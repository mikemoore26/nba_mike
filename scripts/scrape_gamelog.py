import argparse
from random import randint

import pandas as pd

from nba_scraper.config import PLAYERS_CSV
from nba_scraper.gamelogs import get_gamelogs


def _resolve_years(args, players: pd.DataFrame) -> tuple[int, int]:
    available_years = sorted(pd.to_numeric(players["year"], errors="coerce").dropna().astype(int).unique())
    if not available_years:
        raise ValueError("No valid 'year' values found in players.csv")

    latest_year = max(available_years)

    if args.daily:
        return latest_year, latest_year

    if args.year is not None:
        return args.year, args.year

    start_year = args.start_year if args.start_year is not None else latest_year
    end_year = args.end_year if args.end_year is not None else start_year

    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) cannot be greater than end_year ({end_year})")

    return start_year, end_year


def main():
    parser = argparse.ArgumentParser(description="Scrape Basketball Reference gamelogs")
    parser.add_argument("--year", type=int, help="Scrape a single season year (e.g. 2026)")
    parser.add_argument("--start-year", type=int, help="Start season year for range scrape")
    parser.add_argument("--end-year", type=int, help="End season year for range scrape")
    parser.add_argument("--daily", action="store_true", help="Scrape only the latest/current season in players.csv")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--force-refresh", action="store_true", help="Rebuild the target season partition from scratch")
    parser.add_argument("--delay-min", type=float, default=2.5)
    parser.add_argument("--delay-max", type=float, default=5.0)

    args = parser.parse_args()

    if args.delay_min <= 0 or args.delay_max <= 0:
        raise ValueError("delay-min and delay-max must be > 0")
    if args.delay_min > args.delay_max:
        raise ValueError("delay-min cannot be greater than delay-max")

    players = pd.read_csv(PLAYERS_CSV)
    if "year" not in players.columns:
        raise KeyError("players.csv must contain a 'year' column")

    start_year, end_year = _resolve_years(args, players)

    total_players = 0
    for year in range(start_year, end_year + 1):
        players_df = players[pd.to_numeric(players["year"], errors="coerce") == year].copy()
        total_players += len(players_df)

    print(
        f"[INFO] Starting gamelog scrape | start_year={start_year} | "
        f"end_year={end_year} | total_players={total_players} | "
        f"daily={args.daily} | debug={args.debug} | force_refresh={args.force_refresh}"
    )

    processed = 0

    for year in range(start_year, end_year + 1):
        players_df = players[pd.to_numeric(players["year"], errors="coerce") == year].copy()

        if players_df.empty:
            print(f"[WARN] No players found for season {year}; skipping.")
            continue

        delay = randint(int(args.delay_min * 10), int(args.delay_max * 10)) / 10.0

        print(
            f"[INFO] Scraping season {year} | players={len(players_df)} | "
            f"progress={processed}/{total_players} | delay={delay:.1f}s"
        )

        get_gamelogs(
            data=players_df,
            year=year,
            debug=args.debug,
            delay=delay,
            force_refresh=args.force_refresh,
        )

        processed += len(players_df)

    print(f"[DONE] Finished gamelog scrape | processed={processed}")


if __name__ == "__main__":
    main()