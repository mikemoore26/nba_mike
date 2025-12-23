from bs4 import BeautifulSoup, Comment
import pandas as pd
import os
import time

from .browser import get_page_source
from .config import BASE_URL, DATA_DIR, TEAMS_CSV, PLAYERS_CSV
from .teams import get_teams
from .progress import load_progress, save_progress

def parse_roster_table(html: str, team: str, year: int) -> list:
    soup = BeautifulSoup(html, "html.parser")

    roster_table = soup.find("table", {"id": "roster"})
    if not roster_table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if 'id="roster"' in c:
                roster_table = BeautifulSoup(c, "html.parser").find("table", id="roster")
                break

    if not roster_table:
        print(f"[WARN] No roster table for team {team} : {year}. Skipping.")
        return []

    rows = roster_table.find("tbody").find_all("tr")

    players = []
    for row in rows:
        player_cell = row.find("td", {"data-stat": "player"})
        if not player_cell:
            continue

        player = {}
        player["name"]  = player_cell.get_text(strip=True)
        player["pos"]   = row.find("td", {"data-stat": "pos"}).get_text(strip=True)
        player["team"]  = team
        player["number"] = row.find("th", {"data-stat": "number"}).get_text(strip=True)
        player["href"]  = BASE_URL + player_cell.find("a")["href"]
        player["birth_date"] = row.find("td", {"data-stat": "birth_date"}).get_text(strip=True)

        players.append(player)

    return players


def initialize_players(start_year: int = 2024, end_year: int = 2025):
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        teams_df = pd.read_csv(TEAMS_CSV)
    except FileNotFoundError:
        teams_df = get_teams()

    teams = teams_df["href"].tolist()

    done = load_progress()
    print(f"Loaded {len(done)} completed (year, team) combos from progress file.")

    if os.path.exists(PLAYERS_CSV):
        players_df = pd.read_csv(PLAYERS_CSV)
        players = players_df.to_dict(orient="records")
        print(f"Loaded {len(players)} existing players from {PLAYERS_CSV}.")
    else:
        players = []

    for year in range(start_year, end_year):
        for i, team in enumerate(teams):
            if (year, team) in done:
                print(f"[SKIP] Already processed {team} {year}")
                continue

            print(f"Processing team {i+1}/{len(teams)}: {team} for year {year}")
            url = f"{BASE_URL}/teams/{team}/{year}.html"
            html = get_page_source(url)

            if not html:
                print(f"[WARN] Empty HTML for {team} {year}. Skipping.")
                done.add((year, team))
                save_progress(done)
                continue

            new_players = parse_roster_table(html, team, year)
            players.extend(new_players)

            pd.DataFrame(players).drop_duplicates(subset=['href']).to_csv(PLAYERS_CSV, index=False)
            done.add((year, team))
            save_progress(done)

            time.sleep(3)

    print(f"Finished. Total players: {len(players)}")
