from pathlib import Path
import pandas as pd

from nba_scraper.storage import gamelog_dataset_root, write_gamelog_part

GAMELOG_DIR = Path("data/gamelogs")  # adjust to your DATA_DIR / "gamelogs"

def migrate_one_year(year: int):
    csv_path = GAMELOG_DIR / f"gamelogs_{year}.csv"
    if not csv_path.exists():
        print("missing:", csv_path)
        return

    root = gamelog_dataset_root(GAMELOG_DIR)

    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        # ensure season column exists / correct
        chunk["season"] = year
        write_gamelog_part(root, year, chunk)
        total += len(chunk)
        print(f"wrote chunk: {len(chunk)} (total {total})")

    print("done:", year, "rows:", total, "->", root / f"season={year}")

if __name__ == "__main__":
    for year in range(2021, 2027):
        migrate_one_year(year)
