from pathlib import Path
import pandas as pd

ROOT = Path("./data/gamelogs/gamelogs_parquet")

def repair_parquet_schema():
    files = sorted(ROOT.rglob("*.parquet"))
    print("files:", len(files))
    for p in files:
        df = pd.read_parquet(p)

        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce")
            df = df.dropna(subset=["season"])
            df["season"] = df["season"].astype("int32")

        tmp = p.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="snappy")
        tmp.replace(p)

    print("done")

if __name__ == "__main__":
    repair_parquet_schema()
