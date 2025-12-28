# nba_scraper/config.py

from pathlib import Path

BASE_URL = "https://www.basketball-reference.com"

# project root (nba_stuff/)
ROOT_DIR = Path(__file__).resolve().parents[1]

# data directory
DATA_DIR = ROOT_DIR / "data"

# subdirs
PROGRESS_DIR = DATA_DIR / "progress"
GAMELOG_DIR = DATA_DIR / "gamelogs"

LOG_DIR = DATA_DIR / "logs"

# common files
PROGRESS_FILE = PROGRESS_DIR / "progress.json"
PLAYERS_CSV = DATA_DIR / "players.csv"
TEAMS_CSV  = DATA_DIR / "teams.csv"
