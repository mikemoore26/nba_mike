from bs4 import BeautifulSoup
import pandas as pd
import os
from .browser import get_page_source
from .config import BASE_URL, DATA_DIR, TEAMS_CSV

def get_teams():
    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{BASE_URL}/teams/"
    html = get_page_source(url)
    if not html:
        raise ValueError("Could not retrieve the page source.")

    soup = BeautifulSoup(html, "html.parser")
    teams_table = soup.find("table", {"id": "teams_active"})
    if not teams_table:
        raise ValueError("Could not find teams table.")

    rows = teams_table.find("tbody").find_all("tr")
    teams = []

    for row in rows:
        name_cell = row.find("th", {"data-stat": "franch_name"})
        if not name_cell:
            continue

        link = name_cell.find("a")
        if not link:
            continue

        team = {
            "name": link.get_text(strip=True),
            "href": link["href"].split("/")[2],  # team code
        }
        teams.append(team)


    return teams
