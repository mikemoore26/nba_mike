# nba_scraper/rosters.py

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment
from pandas.errors import EmptyDataError

from nba_scraper.browser import get_page_source
from nba_scraper.config import BASE_URL, DATA_DIR, TEAMS_CSV, PLAYERS_CSV, PROGRESS_DIR
from nba_scraper.teams import get_teams

# ---------------------------------------------------------------------------
# Simple logging
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO"):
    print(f"[{level}] {msg}")


# ---------------------------------------------------------------------------
# Paths / progress
# ---------------------------------------------------------------------------

ROSTER_PROGRESS_FILE: Path = PROGRESS_DIR / "progress_rosters.json"


def load_roster_progress() -> set[tuple[int, str]]:
    if not ROSTER_PROGRESS_FILE.exists():
        return set()

    with ROSTER_PROGRESS_FILE.open("r") as f:
        data = json.load(f)

    return {(int(year), team) for year, team in data}


def save_roster_progress(done_set: set[tuple[int, str]]) -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    data = [[year, team] for (year, team) in sorted(done_set)]
    with ROSTER_PROGRESS_FILE.open("w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cell_text(row, data_stat: str, header: bool = False) -> str:
    tag = "th" if header else "td"
    cell = row.find(tag, {"data-stat": data_stat})
    return cell.get_text(strip=True) if cell else ""


# ---------------------------------------------------------------------------
# Parse roster page
# ---------------------------------------------------------------------------

def parse_roster_table(html: str, team: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    roster_table = soup.find("table", {"id": "roster"})
    if not roster_table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if 'id="roster"' in c:
                roster_table = BeautifulSoup(c, "html.parser").find("table", id="roster")
                break

    if not roster_table:
        log(f"No roster table for team {team} {year}. Skipping.", "WARN")
        return []

    rows = roster_table.find("tbody").find_all("tr")

    players: list[dict] = []
    for row in rows:
        player_cell = row.find("td", {"data-stat": "player"})
        if not player_cell:
            continue

        name = player_cell.get_text(strip=True)
        href_tag = player_cell.find("a")

        players.append({
            "name": name,
            "year": year,
            "pos": _get_cell_text(row, "pos"),
            "team": team,
            "number": _get_cell_text(row, "number", header=True),
            "birth_date": _get_cell_text(row, "birth_date"),
            "href": BASE_URL + href_tag["href"] if href_tag else "",
        })

    return players


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def initialize_players(
    start_year: int = 2024,
    end_year: int = 2025,
    delay: float = 3.0,
) -> list[dict]:

    DATA_DIR.mkdir(exist_ok=True)
    PROGRESS_DIR.mkdir(exist_ok=True)

    if TEAMS_CSV.exists():
        teams_df = pd.read_csv(TEAMS_CSV)
    else:
        teams_df = get_teams()
        teams_df.to_csv(TEAMS_CSV, index=False)

    teams = teams_df["href"].tolist()

    done = load_roster_progress()
    log(f"Loaded {len(done)} completed (year, team) combos from roster progress.")

    players: list[dict] = []
    seen_keys: set[tuple[str, int]] = set()
    file_initialized = False

    if PLAYERS_CSV.exists():
        try:
            existing_df = pd.read_csv(PLAYERS_CSV)
            players = existing_df.to_dict(orient="records")
            log(f"Loaded {len(players)} existing players from {PLAYERS_CSV}.")

            # Track (href, year) pairs we've already seen
            for p in players:
                href = p.get("href")
                year_val = p.get("year")
                if href and year_val is not None:
                    seen_keys.add((href, int(year_val)))

            if not existing_df.empty:
                file_initialized = True
        except EmptyDataError:
            log(f"{PLAYERS_CSV} exists but is empty. Starting fresh.", "WARN")

    for year in range(start_year, end_year + 1):
        log(f"Season {year} starting...")

        for i, team in enumerate(teams, start=1):
            if (year, team) in done:
                log(f"{team} {year} already processed", "SKIP")
                continue

            log(f"({i}/{len(teams)}) Scraping roster for {team} {year}")
            url = f"{BASE_URL}/teams/{team}/{year}.html"
            html = get_page_source(url)

            if not html:
                log(f"Empty HTML for {team} {year}. Skipping.", "WARN")
                done.add((year, team))
                save_roster_progress(done)
                continue

            new_players = parse_roster_table(html, team, year)

            if new_players:
                unique_new: list[dict] = []
                for p in new_players:
                    key = (p["href"], p["year"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_new.append(p)

                if unique_new:
                    players.extend(unique_new)
                    df = pd.DataFrame(unique_new)

                    # Append only new rows; write header once
                    mode = "a" if file_initialized else "w"
                    df.to_csv(PLAYERS_CSV, mode=mode, index=False, header=not file_initialized)
                    file_initialized = True

            done.add((year, team))
            save_roster_progress(done)

            time.sleep(delay)

    log(f"Finished roster scraping. Total players: {len(players)}")
    return players
