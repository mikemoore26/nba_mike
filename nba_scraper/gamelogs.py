from __future__ import annotations

import json
import shutil
import time
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment

from nba_scraper.config import DATA_DIR, PROGRESS_DIR, PLAYERS_CSV
from nba_scraper.browser import fetch_html, log
from nba_scraper.storage import (
    gamelog_dataset_root,
    load_seen_keys_from_parquet_dataset,
    write_gamelog_part,
)
from model_training.utils.team_codes import norm_team


GAMELOG_DIR: Path = DATA_DIR / "gamelogs"
GAMELOG_PROGRESS_FILE: Path = PROGRESS_DIR / "progress_gamelogs.json"


def _key(season: int, href: str) -> str:
    return f"{season}|{href}"


def load_gamelog_progress() -> dict[str, str]:
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
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GAMELOG_PROGRESS_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(GAMELOG_PROGRESS_FILE)


def clear_progress_for_season(season: int, progress: dict[str, str]) -> dict[str, str]:
    prefix = f"{season}|"
    return {k: v for k, v in progress.items() if not k.startswith(prefix)}


def is_blocked_html(html: str) -> bool:
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


def parse_gamelog_page(html: str, player_name: str, href: str, season: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", id="player_game_log_reg")

    if not table:
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

    def get_td(row, stat_name: str) -> str:
        cell = row.find("td", {"data-stat": stat_name})
        return cell.get_text(strip=True) if cell else ""

    def get_th_or_td(row, stat_name: str) -> str:
        cell = row.find(["th", "td"], {"data-stat": stat_name})
        return cell.get_text(strip=True) if cell else ""

    for row in rows:
        if row.get("data-row") is None:
            continue

        team_raw = get_td(row, "team_name_abbr")
        opp_raw = get_td(row, "opp_name_abbr")

        # Basketball-Reference often stores date in a TH cell, not TD.
        date_val = get_th_or_td(row, "date_game")
        if not date_val:
            date_val = get_th_or_td(row, "date")

        mp_val = get_td(row, "mp")

        games.append(
            {
                "player": player_name,
                "href": href,
                "season": season,
                "date": date_val,
                "team": norm_team(team_raw),
                "opp": norm_team(opp_raw),
                "home_away": get_td(row, "game_location"),
                "result": get_td(row, "game_result"),
                "gs": get_td(row, "gs"),
                "mp": mp_val,
                "fg": get_td(row, "fg"),
                "fga": get_td(row, "fga"),
                "fg3": get_td(row, "fg3"),
                "fg3a": get_td(row, "fg3a"),
                "ft": get_td(row, "ft"),
                "fta": get_td(row, "fta"),
                "orb": get_td(row, "orb"),
                "drb": get_td(row, "drb"),
                "trb": get_td(row, "trb"),
                "ast": get_td(row, "ast"),
                "stl": get_td(row, "stl"),
                "blk": get_td(row, "blk"),
                "tov": get_td(row, "tov"),
                "pf": get_td(row, "pf"),
                "pts": get_td(row, "pts"),
            }
        )

    # quick sanity signal if parser is failing
    if games:
        sample_mp = [g["mp"] for g in games[:5]]
        log(f"[DEBUG] {player_name} season={season} sample mp values: {sample_mp}", "DEBUG")

    return games


def get_gamelogs(
    data: pd.DataFrame | None = None,
    year: int = 2026,
    debug: bool = False,
    delay: float = 5.0,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Scrape Basketball-Reference player gamelogs into append-only parquet parts.

    force_refresh=True:
      - clears existing parquet season partition
      - clears season progress entries
      - ignores seen_keys/progress short-circuiting
      - fully rebuilds the season
    """
    DATA_DIR.mkdir(exist_ok=True)
    GAMELOG_DIR.mkdir(exist_ok=True)
    PROGRESS_DIR.mkdir(exist_ok=True)

    if data is None:
        data = pd.read_csv(PLAYERS_CSV)

    if "href" in data.columns:
        data = data.sort_values("href", kind="stable")

    progress = load_gamelog_progress()
    today = date.today().isoformat()
    log(f"Loaded {len(progress)} progress entries from gamelog progress file.")

    def scraped_today(season: int, href: str) -> bool:
        return progress.get(_key(season, href)) == today

    games: list[dict] = []
    root = gamelog_dataset_root(GAMELOG_DIR)
    season_root = root / f"season={year}"

    if force_refresh:
        if season_root.exists():
            shutil.rmtree(season_root)
            log(f"[FORCE_REFRESH] Removed existing season partition: {season_root}")
        progress = clear_progress_for_season(year, progress)
        save_gamelog_progress(progress)
        seen_keys: set[tuple[str, int, str]] = set()
        log(f"[FORCE_REFRESH] Cleared progress entries for season={year}")
    else:
        seen_keys = load_seen_keys_from_parquet_dataset(root, year)
        log(f"Loaded {len(seen_keys)} existing parquet keys for season={year}.")

    driver = None

    try:
        total_players = len(data)

        for i, row in enumerate(data.itertuples(index=False), start=1):
            href = getattr(row, "href")
            player_name = getattr(row, "name")

            log(f"{i}/{total_players} Processing: {player_name}")

            if (not force_refresh) and scraped_today(year, href):
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

            new_games = parse_gamelog_page(
                html=html,
                player_name=player_name,
                href=href,
                season=year,
            )

            if new_games:
                unique_new: list[dict] = []
                for g in new_games:
                    key = (g["href"], g["season"], g["date"])
                    if force_refresh or key not in seen_keys:
                        seen_keys.add(key)
                        unique_new.append(g)

                if unique_new:
                    df = pd.DataFrame(unique_new)
                    written = write_gamelog_part(root, year, df)
                    log(f"Appended {len(df)} rows -> {written}")

                    progress[_key(year, href)] = today
                    save_gamelog_progress(progress)

                    games.extend(unique_new)
                else:
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