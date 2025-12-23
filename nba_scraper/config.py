import os

BASE_URL = "https://www.basketball-reference.com"

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
PLAYERS_CSV = os.path.join(DATA_DIR, "players.csv")
TEAMS_CSV = os.path.join(DATA_DIR, "teams.csv")
