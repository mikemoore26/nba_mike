# nba_scraper/gamelogs.py

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment
from pandas.errors import EmptyDataError

from nba_scraper.browser import get_page_source
from nba_scraper.config import DATA_DIR, PROGRESS_DIR, PLAYERS_CSV

# --- Simple logging ----------------------------------------------------------

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")


# --- Paths -------------------------------------------------------------------

GAMELOG_DIR: Path = DATA_DIR / "gamelogs"
GAMELOG_PROGRESS_FILE: Path = PROGRESS_DIR / "progress_gamelogs.json"


# --- Progress helpers --------------------------------------------------------

def load_gamelog_progress() -> set[tuple[int, str]]:
    """
    Returns a set of (season, href) tuples we've already scraped.
    """
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
    """
    Parse a Basketball-Reference game log page like:
    https://www.basketball-reference.com/players/c/chrisma01/gamelog/2017
    """
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", id="player_game_log_reg")

    # Fallback if hidden in HTML comments
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

        game = {
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
        }

        games.append(game)

    return games


# --- High level scrape function ---------------------------------------------

def get_gamelogs(
    data: pd.DataFrame | None = None,
    year: int = 2026,
    debug: bool = False,
    delay: float = 1.0,
) -> list[dict]:
    """
    Scrape gamelogs for a given season and save to data/gamelogs/gamelogs_{year}.csv
    """
    DATA_DIR.mkdir(exist_ok=True)
    GAMELOG_DIR.mkdir(exist_ok=True)
    PROGRESS_DIR.mkdir(exist_ok=True)

    # Load player list if not provided
    if data is None:
        data = pd.read_csv(PLAYERS_CSV)

    done = load_gamelog_progress()
    log(f"Loaded {len(done)} completed (season, href) combos from gamelog progress.")

    out_path = GAMELOG_DIR / f"gamelogs_{year}.csv"

    games: list[dict] = []
    seen_keys: set[tuple[str, int, str]] = set()
    file_initialized = False

    # If CSV already exists, keep what we have and populate seen_keys
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

        html = get_page_source(url)
        if not html:
            log(f"Empty HTML for {href} {year}. Skipping.", "WARN")
            done.add((year, href))
            save_gamelog_progress(done)
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
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_new.append(g)

            if unique_new:
                games.extend(unique_new)
                df = pd.DataFrame(unique_new)

                # Append only new rows; write header once
                mode = "a" if file_initialized else "w"
                df.to_csv(out_path, mode=mode, index=False, header=not file_initialized)
                file_initialized = True

        done.add((year, href))
        save_gamelog_progress(done)

        time.sleep(delay)

        if debug and new_games:
            log("[DEBUG] Stopping after first player with data.", "DEBUG")
            break

    log(f"Finished gamelog scraping for season {year}. Total games: {len(games)}")
    return games
