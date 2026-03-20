from __future__ import annotations

import pandas as pd
from pathlib import Path
from model_training.config import PATH_GAMLOGS_COMBINED


PLAYERS_TO_CHECK = [
    "JARRETT ALLEN",
    "RUDY GOBERT",
    "NIKOLA JOKIĆ",
    "WENDELL CARTER JR.",
    "DESMOND BANE",
    "SAM MERRILL",
]


def main():

    output_path = Path("debug_fg3_report.txt")

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("FG3 HISTORY DEBUG REPORT\n")
        f.write("=" * 80 + "\n\n")

        # ---------------------------------------------------
        # Load dataset
        # ---------------------------------------------------

        f.write("Loading dataset:\n")
        f.write(str(PATH_GAMLOGS_COMBINED) + "\n\n")

        df = pd.read_csv(PATH_GAMLOGS_COMBINED)

        # normalize names
        df["player"] = df["player"].astype(str).str.upper()

        # ensure date column works
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        cols = [
            "date",
            "player",
            "team",
            "opp",
            "fg3a",
            "fg3",
            "fga",
            "mp_minutes",
            "pts",
            "usage",
        ]

        cols = [c for c in cols if c in df.columns]

        # ---------------------------------------------------
        # Inspect individual players
        # ---------------------------------------------------

        for player in PLAYERS_TO_CHECK:

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(player + "\n")
            f.write("=" * 80 + "\n")

            sub = df[df["player"] == player].copy()

            if sub.empty:
                f.write("No rows found\n")
                continue

            sub = sub.sort_values("date")

            f.write(sub[cols].to_string(index=False))
            f.write("\n\n")

            if "fg3a" in sub.columns:
                f.write("3PA SUMMARY\n")
                f.write(str(sub["fg3a"].describe()))
                f.write("\n\n")

        # ---------------------------------------------------
        # Suspicious big-man 3PA rows
        # ---------------------------------------------------

        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("SUSPICIOUS BIG-MAN FG3A ROWS (>=3 ATTEMPTS)\n")
        f.write("=" * 80 + "\n\n")

        suspicious_players = [
            "JARRETT ALLEN",
            "RUDY GOBERT",
            "NIKOLA JOKIĆ",
            "WENDELL CARTER JR.",
            "JAKOB POELTL",
            "EVAN MOBLEY",
        ]

        sus = df[df["player"].isin(suspicious_players)].copy()

        if "fg3a" in sus.columns:
            sus = sus[sus["fg3a"] >= 3]

            if not sus.empty:
                f.write(
                    sus[cols]
                    .sort_values(["player", "date"])
                    .to_string(index=False)
                )
            else:
                f.write("No suspicious rows found\n")

        else:
            f.write("fg3a column not found\n")

    print("\nDebug report saved to:")
    print(output_path)


if __name__ == "__main__":
    main()