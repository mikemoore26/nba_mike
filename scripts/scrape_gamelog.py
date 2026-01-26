# scripts/scrape_gamelog.py

import argparse
import pandas as pd

from nba_scraper.gamelogs import get_gamelogs
from nba_scraper.config import PLAYERS_CSV

from random import randint


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--year", type=int, default=2026, help="Season year (e.g. 2026)")
    # parser.add_argument("--debug", action="store_true")
    # args = parser.parse_args()

    end_year = 2026
    debug = False
    start_year = end_year 

    players = pd.read_csv(PLAYERS_CSV)
    i = 0 
    x = len(players) * (end_year - start_year + 1)
    #get_gamelogs(data=players, year=args.year, debug=args.debug)
    
    for year in range(start_year, end_year + 1):
        players_df = players[players["year"] == year]
        print(f"Starting gamelog scrape for season {year} ({i}/{x})")  
        get_gamelogs(data=players_df, year=year, debug=debug, delay=randint(5,10)/2)
        i += len(players_df)


if __name__ == "__main__":
    main()
