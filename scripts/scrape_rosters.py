from nba_scraper.rosters import initialize_players

if __name__ == "__main__":
    # example: last 7 years
    end_year = 2026
    start_year = end_year - 7
    initialize_players(start_year, end_year)
