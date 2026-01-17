from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment
from pandas.errors import EmptyDataError

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from nba_scraper.config import DATA_DIR, PROGRESS_DIR, PLAYERS_CSV
from nba_scraper.browser import create_driver, wait_for_any, log

# --- Paths -------------------------------------------------------------------

GAMELOG_DIR: Path = DATA_DIR / "gamelogs"
GAMELOG_PROGRESS_FILE: Path = PROGRESS_DIR / "progress_gamelogs.json"


# --- Progress helpers --------------------------------------------------------

def load_gamelog_progress() -> set[tuple[int, str]]:
    if not GAMELOG_PROGRESS_FILE.exists():
        return set()

    with GAMELOG_PROGRESS_FILE.open("r") as f:
        data = json.load(f)

    return {(int(season), href) for season, href in data}


def save_gamelog_progress(done_set: set[tuple[int, str]]) -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    data = [[season, href] for (season, href) in sorted(done_set)]
    with GAMELOG_PROGRESS_FILE.open("w") as f:
        json.dump(data, f)


# --- Parsing a single gamelog page ------------------------------------------

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

    rows = table.find("tbody").find_all("tr")
    games: list[dict] = []

    def get_stat(row, stat_name: str) -> str:
        cell = row.find("td", {"data-stat": stat_name})
        return cell.get_text(strip=True) if cell else ""

    for row in rows:
        if row.get("data-row") is None:
            continue

        games.append({
            "player":  player_name,
            "href":    href,
            "season":  season,

            "date":        get_stat(row, "date"),
            "team":        get_stat(row, "team_name_abbr"),
            "opp":         get_stat(row, "opp_name_abbr"),
            "home_away":   get_stat(row, "game_location"),
            "result":      get_stat(row, "game_result"),
            "gs":          get_stat(row, "gs"),

            "mp":   get_stat(row, "mp"),

            "fg":   get_stat(row, "fg"),
            "fga":  get_stat(row, "fga"),
            "fg3":  get_stat(row, "fg3"),
            "fg3a": get_stat(row, "fg3a"),
            "ft":   get_stat(row, "ft"),
            "fta":  get_stat(row, "fta"),

            "orb":  get_stat(row, "orb"),
            "drb":  get_stat(row, "drb"),
            "trb":  get_stat(row, "trb"),

            "ast":  get_stat(row, "ast"),
            "stl":  get_stat(row, "stl"),
            "blk":  get_stat(row, "blk"),
            "tov":  get_stat(row, "tov"),
            "pf":   get_stat(row, "pf"),
            "pts":  get_stat(row, "pts"),
        })

    return games


# --- High level scrape function ---------------------------------------------

def get_gamelogs(
    data: pd.DataFrame | None = None,
    year: int = 2026,
    debug: bool = False,
    delay: float = 1.0,
) -> list[dict]:

    DATA_DIR.mkdir(exist_ok=True)
    GAMELOG_DIR.mkdir(exist_ok=True)
    PROGRESS_DIR.mkdir(exist_ok=True)

    if data is None:
        data = pd.read_csv(PLAYERS_CSV)

    done = load_gamelog_progress()
    log(f"Loaded {len(done)} completed (season, href) combos from gamelog progress.")

    out_path = GAMELOG_DIR / f"gamelogs_{year}.csv"

    games: list[dict] = []
    seen_keys: set[tuple[str, int, str]] = set()
    file_initialized = False

    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            games = existing_df.to_dict(orient="records")
            log(f"Loaded {len(games)} existing game rows from {out_path}.")

            for g in games:
                href_val = g.get("href")
                season_val = g.get("season")
                date_val = g.get("date")
                if href_val and season_val is not None and date_val:
                    seen_keys.add((href_val, int(season_val), str(date_val)))

            if not existing_df.empty:
                file_initialized = True
        except EmptyDataError:
            log(f"{out_path} exists but is empty. Starting fresh.", "WARN")

    # Option A: keep driver updated by returning it from fetch_html
    driver = None

    def fetch_html(driver, url: str, retries: int = 3, retry_delay: float = 5.0):
        if driver is None:
            driver = create_driver()

        for attempt in range(1, retries + 1):
            try:
                driver.set_page_load_timeout(25)
                driver.get(url)

                # Wait for table or at least body
                wait_for_any(driver, ["#player_game_log_reg", "body"], timeout=15)

                # Stop extra loading to reduce renderer hangs
                try:
                    driver.execute_script("window.stop();")
                except Exception:
                    pass

                return driver.page_source or "", driver

            except Exception as e:
                msg = str(e)
                log(f"Error loading {url} (attempt {attempt}/{retries}): {e}", "WARN")

                hard_restart = (
                    "HTTPConnectionPool(host='localhost'" in msg
                    or "Read timed out" in msg
                    or "MaxRetryError" in msg
                    or "chrome not reachable" in msg
                    or "disconnected" in msg
                    or "Timed out receiving message from renderer" in msg
                )

                try:
                    driver.execute_script("window.stop();")
                except Exception:
                    pass

                try:
                    driver.quit()
                except Exception:
                    pass

                time.sleep(2)
                driver = create_driver()

                time.sleep(2 if hard_restart else retry_delay)

        return "", driver

    try:
        total_players = len(data)
        for i, row in enumerate(data.itertuples(index=False), start=1):
            log(f"{i}/{total_players}")
            log(f"Processing player: {row.name}")
            href = row.href
            player_name = row.name

            if (year, href) in done:
                log(f"{href} {year} already processed", "SKIP")
                continue

            url = href.replace(".html", f"/gamelog/{year}")
            log(f"Scraping gamelog: {url}")

            html, driver = fetch_html(driver, url)
            if not html:
                log(f"Empty HTML for {href} {year}. Skipping.", "WARN")
                done.add((year, href))
                save_gamelog_progress(done)
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
                    games.extend(unique_new)
                    df = pd.DataFrame(unique_new)

                    mode = "a" if file_initialized else "w"
                    df.to_csv(out_path, mode=mode, index=False, header=not file_initialized)
                    file_initialized = True

            done.add((year, href))
            save_gamelog_progress(done)

            time.sleep(delay)

            if debug and new_games:
                log("[DEBUG] Stopping after first player with data.", "DEBUG")
                break

    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass

    log(f"Finished gamelog scraping for season {year}. Total games: {len(games)}")
    return games
