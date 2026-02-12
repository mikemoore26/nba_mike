# nba_scraper/gamelogs.py
from __future__ import annotations

import json
from logging import root
import time
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment
from pandas.errors import EmptyDataError

from nba_scraper.config import DATA_DIR, PROGRESS_DIR, PLAYERS_CSV
from nba_scraper.browser import fetch_html, log

from nba_scraper.storage import (
    gamelog_dataset_root,
    load_seen_keys_from_parquet_dataset,
    write_gamelog_part,
)



# --- Paths -------------------------------------------------------------------

GAMELOG_DIR: Path = DATA_DIR / "gamelogs"
GAMELOG_PROGRESS_FILE: Path = PROGRESS_DIR / "progress_gamelogs.json"


# --- Progress helpers --------------------------------------------------------

def _key(season: int, href: str) -> str:
    return f"{season}|{href}"


def load_gamelog_progress() -> dict[str, str]:
    """
    Returns dict: "season|href" -> "YYYY-MM-DD"
    Backward compatible with old format: [[season, href], ...]
    """
    if not GAMELOG_PROGRESS_FILE.exists():
        return {}

    with GAMELOG_PROGRESS_FILE.open("r") as f:
        data = json.load(f)

    if isinstance(data, list):
        today = date.today().isoformat()
        return {_key(int(season), href): today for season, href in data}

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    return {}


def save_gamelog_progress(progress: dict[str, str]) -> None:
    """
    Atomic write (crash-safe).
    """
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GAMELOG_PROGRESS_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(GAMELOG_PROGRESS_FILE)


# --- Block / throttle detection ---------------------------------------------

def is_blocked_html(html: str) -> bool:
    """
    Very lightweight heuristic detection.
    """
    if not html:
        return True

    h = html.lower()
    signals = [
        "captcha",
        "verify you are human",
        "unusual traffic",
        "access denied",
        "temporarily blocked",
        "request blocked",
        "cloudflare",
        "incapsula",
        "automated queries",
    ]
    return any(s in h for s in signals)


# --- Parsing a single gamelog page ------------------------------------------

def parse_gamelog_page(html: str, player_name: str, href: str, season: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", id="player_game_log_reg")

    if not table:
        # Basketball-Reference sometimes hides tables inside HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if 'id="player_game_log_reg"' in c:
                table = BeautifulSoup(c, "html.parser").find("table", id="player_game_log_reg")
                break

    if not table:
        log(f"No game log table for {href} {season}", "WARN")
        return []

    tbody = table.find("tbody")
    if not tbody:
        log(f"No tbody for game log table {href} {season}", "WARN")
        return []

    rows = tbody.find_all("tr")
    games: list[dict] = []

    def get_stat(row, stat_name: str) -> str:
        cell = row.find("td", {"data-stat": stat_name})
        return cell.get_text(strip=True) if cell else ""

    for row in rows:
        # skip separators / headers within tbody
        if row.get("data-row") is None:
            continue

        games.append(
            {
                "player": player_name,
                "href": href,
                "season": season,
                "date": get_stat(row, "date"),
                "team": get_stat(row, "team_name_abbr"),
                "opp": get_stat(row, "opp_name_abbr"),
                "home_away": get_stat(row, "game_location"),
                "result": get_stat(row, "game_result"),
                "gs": get_stat(row, "gs"),
                "mp": get_stat(row, "mp"),
                "fg": get_stat(row, "fg"),
                "fga": get_stat(row, "fga"),
                "fg3": get_stat(row, "fg3"),
                "fg3a": get_stat(row, "fg3a"),
                "ft": get_stat(row, "ft"),
                "fta": get_stat(row, "fta"),
                "orb": get_stat(row, "orb"),
                "drb": get_stat(row, "drb"),
                "trb": get_stat(row, "trb"),
                "ast": get_stat(row, "ast"),
                "stl": get_stat(row, "stl"),
                "blk": get_stat(row, "blk"),
                "tov": get_stat(row, "tov"),
                "pf": get_stat(row, "pf"),
                "pts": get_stat(row, "pts"),
            }
        )

    return games


# --- High level scrape function ---------------------------------------------

def get_gamelogs(
    data: pd.DataFrame | None = None,
    year: int = 2026,
    debug: bool = False,
    delay: float = 5.0,
) -> list[dict]:
    """
    Production-ish constraints:
      - append-only CSV
      - idempotent reruns (dedupe by (href, season, date))
      - progress means LAST SUCCESS date (not "attempted")
      - chunked existing-key load (fast)
      - centralized fetch_html() in browser.py
    """
    DATA_DIR.mkdir(exist_ok=True)
    GAMELOG_DIR.mkdir(exist_ok=True)
    PROGRESS_DIR.mkdir(exist_ok=True)

    if data is None:
        data = pd.read_csv(PLAYERS_CSV)

    # determinism / reproducibility: stable order
    if "href" in data.columns:
        data = data.sort_values("href", kind="stable")

    progress = load_gamelog_progress()
    today = date.today().isoformat()
    log(f"Loaded {len(progress)} progress entries from gamelog progress file.")

    def scraped_today(season: int, href: str) -> bool:
        # progress stores LAST SUCCESS date
        return progress.get(_key(season, href)) == today

    out_path = GAMELOG_DIR / f"gamelogs_{year}.csv"

    games: list[dict] = []
    seen_keys: set[tuple[str, int, str]] = set()
    file_initialized = False

    # Load existing keys only (chunked) so reruns are idempotent without OOM risk
    root = gamelog_dataset_root(GAMELOG_DIR)
    seen_keys = load_seen_keys_from_parquet_dataset(root, year)
    log(f"Loaded {len(seen_keys)} existing parquet keys for season={year}.")

    driver = None

    try:
        total_players = len(data)

        for i, row in enumerate(data.itertuples(index=False), start=1):
            href = getattr(row, "href")
            player_name = getattr(row, "name")

            log(f"{i}/{total_players}  Processing: {player_name}")

            if scraped_today(year, href):
                log(f"{href} {year} already scraped today", "SKIP")
                continue

            url = href.replace(".html", f"/gamelog/{year}")
            log(f"Scraping gamelog: {url}")

            html, driver = fetch_html(
                driver,
                url,
                retries=3,
                retry_delay=max(5.0, float(delay)),
                wait_for_css_any=["#player_game_log_reg", "body"],
                timeout=15.0,
            )

            if is_blocked_html(html):
                log(f"Blocked/throttled HTML detected for {href} {year}. Backing off + skipping.", "WARN")
                time.sleep(30 + (5 * (i % 3)))
                continue

            if not html:
                log(f"Empty HTML for {href} {year}. Skipping.", "WARN")
                time.sleep(delay + (0.5 * (i % 3)))
                continue

            new_games = parse_gamelog_page(html=html, player_name=player_name, href=href, season=year)

            if new_games:
                unique_new: list[dict] = []
                for g in new_games:
                    key = (g["href"], g["season"], g["date"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_new.append(g)

                if unique_new:
                    df = pd.DataFrame(unique_new)
                    written = write_gamelog_part(root, year, df)
                    log(f"Appended {len(df)} rows -> {written}")

                    file_initialized = True

                    # ✅ progress means success, after durable write
                    progress[_key(year, href)] = today
                    save_gamelog_progress(progress)

                    games.extend(unique_new)
                else:
                    # nothing new, still a "success" scrape
                    progress[_key(year, href)] = today
                    save_gamelog_progress(progress)
            else:
                log(f"No rows parsed for {href} {year}. Not updating progress.", "WARN")

            time.sleep(delay + (0.5 * (i % 3)))

            if debug and new_games:
                log("[DEBUG] Stopping after first player with data.", "DEBUG")
                break

    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass

    log(f"Finished gamelog scraping for season {year}. Total new rows appended this run: {len(games)}")
    return games
