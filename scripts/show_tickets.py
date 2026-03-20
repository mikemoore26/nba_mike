from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd


class TicketViewer:

    def __init__(self):
        # project root
        self.root = Path.cwd()

        # correct results directory
        self.results_dir = self.root / "results"

        self.today = datetime.today().strftime("%Y-%m-%d")

    def _load_ticket(self, name: str) -> pd.DataFrame:

        path = self.results_dir / self.today / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"\nTicket file not found.\nExpected path:\n{path}\n"
            )

        df = pd.read_csv(path)

        cols = ["player", "team", "stat", "pred_mean"]
        cols = [c for c in cols if c in df.columns]

        df = df[cols]

        if "pred_mean" in df.columns:
            df["pred_mean"] = df["pred_mean"].round(2)

        return df

    def show_ticket(self, name: str):

        df = self._load_ticket(name)

        print("\n" + "=" * 50)
        print(name.upper())
        print("=" * 50)

        print(df.to_string(index=False))

        return df

    def show_all(self):

        for name in ["A_safe", "B_balanced", "C_ceiling"]:
            self.show_ticket(name)


if __name__ == "__main__":

    viewer = TicketViewer()
    viewer.show_all()