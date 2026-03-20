# model_training/data_loading.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from model_training.config import GAMELOG_PARQUET_ROOT, PATH_GAMLOGS_COMBINED


# ----------------------------
# Helpers
# ----------------------------
def _mp_series_to_minutes(mp: pd.Series) -> pd.Series:
    """
    Robust vectorized mp 'MM:SS' -> float minutes.
    Handles categorical/pyarrow string/mixed values.
    """
    s = mp.astype("string")
    extracted = s.str.extract(r"^(?P<m>\d+):(?P<s>\d+)$")
    m = pd.to_numeric(extracted["m"], errors="coerce").fillna(0)
    sec = pd.to_numeric(extracted["s"], errors="coerce").fillna(0)
    return m + (sec / 60.0)


def _list_parquet_files(root: Path, seasons: list[int] | None) -> list[Path]:
    files: list[Path] = []
    if seasons:
        for s in seasons:
            part_dir = root / f"season={s}"
            if part_dir.exists():
                files.extend(sorted(part_dir.rglob("*.parquet")))
    else:
        files.extend(sorted(root.rglob("*.parquet")))
    return files


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Ensure required columns exist. Missing ones created as NA.
    This avoids KeyErrors + lets you see if your parquet schema is wrong.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _debug_date_failure(raw_date: pd.Series, df_cols: list[str]) -> None:
    sample = raw_date.dropna().astype(str).head(25).tolist()
    print("[ERROR] All parsed dates are NaT. This means your source 'date' values are not parseable.")
    print("[ERROR] Sample raw date values:", sample)
    print("[ERROR] Columns present:", df_cols)


# ----------------------------
# Loaders
# ----------------------------
def load_gamelogs_parquet(
    *,
    seasons: list[int] | None = None,
    columns: list[str] | None = None,
    root: Path = GAMELOG_PARQUET_ROOT,
) -> pd.DataFrame:
    """
    Parquet dataset loader that is robust to:
      - Windows + pyarrow directory issues
      - schema drift across part files
      - dictionary/categorical season types
    Reads per-file and concatenates (no schema merge).
    """
    if not root.exists():
        raise FileNotFoundError(f"Missing parquet dataset: {root}")

    files = _list_parquet_files(root, seasons)
    if not files:
        raise RuntimeError(f"No parquet files found under {root}")

    dfs: list[pd.DataFrame] = []
    for p in files:
        dfi = pd.read_parquet(str(p), columns=columns)

        # Normalize season drift (dict/categorical -> numeric)
        if "season" in dfi.columns:
            dfi["season"] = pd.to_numeric(dfi["season"], errors="coerce")

        dfs.append(dfi)

    df = pd.concat(dfs, ignore_index=True)

    # Final enforce
    if "season" in df.columns:
        df = df.dropna(subset=["season"])
        df["season"] = df["season"].astype(int)

    return df


def load_gamelogs_legacy_csv(
    *,
    seasons: list[int] | None = None,
    root: Path = Path("./data/gamelogs"),
) -> pd.DataFrame:
    res: list[pd.DataFrame] = []
    for dirpath, _, files in os.walk(root):
        for file in files:
            if not file.endswith(".csv"):
                continue

            if seasons and file not in {f"gamelogs_{s}.csv" for s in seasons}:
                continue

            path = Path(dirpath) / file
            print(f"Processing {path}")
            res.append(pd.read_csv(path))

    if not res:
        raise RuntimeError("No legacy CSV gamelogs found to combine.")

    return pd.concat(res, ignore_index=True)


# ----------------------------
# Builder for training table
# ----------------------------
def build_all_gamelogs_combined(
    *,
    seasons: list[int] | None = None,
    use_parquet: bool = True,
    write_combined_csv: bool = True,
    combined_path: Path = PATH_GAMLOGS_COMBINED,
) -> pd.DataFrame:
    """
    Build the combined modeling table (CSV) used by training.
    Fail-fast if dates are unparseable or output becomes empty.
    """
    need_cols = [
        "player", "date", "season", "mp",
        "team", "opp", "home_away", "result",
        "fg", "fga", "fg3", "fg3a",
        "ft", "fta",
        "orb", "drb", "trb",
        "ast", "stl", "blk",
        "tov", "pf", "pts",
    ]

    # --- load --------------------------------------------------------------
    if use_parquet and GAMELOG_PARQUET_ROOT.exists():
        files = _list_parquet_files(GAMELOG_PARQUET_ROOT, seasons)
        print(f"[INFO] Parquet parts found: {len(files)}")
        df = load_gamelogs_parquet(seasons=seasons, columns=None)  # load full per-file to inspect schema
    else:
        df = load_gamelogs_legacy_csv(seasons=seasons)

    if df.empty:
        raise RuntimeError("Loaded 0 rows from gamelogs source. Check your parquet/csv paths.")

    # Ensure required columns exist, then subset
    df = _ensure_cols(df, need_cols)
    df = df[need_cols].copy()

    # --- basic row validity (do NOT drop 'date' yet; we need diagnostics) ---
    df = df.dropna(subset=["player", "season"])  # player + season must exist

    # --- date parse (FAIL FAST if everything becomes NaT) -------------------
    raw_date = df["date"].astype("string")
    parsed = pd.to_datetime(raw_date, errors="coerce")

    if parsed.isna().all():
        _debug_date_failure(raw_date, list(df.columns))
        raise RuntimeError("All parsed dates are NaT. Fix source 'date' format or column name mismatch.")

    df["date"] = parsed
    df = df.dropna(subset=["date"]).copy()

    # --- season normalize ---------------------------------------------------
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"]).copy()
    df["season"] = df["season"].astype(int)

    # --- compute flags (safe for categorical) -------------------------------
    ha = df["home_away"].astype("string")
    df["is_home"] = ha.ne("@").fillna(True).astype("int8")  # '@' = away

    res = df["result"].astype("string")
    df["is_win"] = res.str.startswith("W").fillna(False).astype("int8")

    # --- mp parsing (do NOT drop mp rows blindly; some are DNP strings) -----
    # Keep mp raw; parse minutes to 0 for non-mm:ss
    df["mp_minutes"] = _mp_series_to_minutes(df["mp"]).astype("float32")
    df["usage"] = (df["mp_minutes"] / 48.0).astype("float32")

    # --- drop cols not needed downstream -----------------------------------
    drop_cols = ["home_away", "result"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- dedupe -------------------------------------------------------------
    df.drop_duplicates(subset=["player", "date", "season"], keep="last", inplace=True)

    # --- final guard --------------------------------------------------------
    if df.empty:
        raise RuntimeError(
            "build_all_gamelogs_combined produced 0 rows after cleaning. "
            "This usually means date parsing or required columns are wrong."
        )

    # --- write --------------------------------------------------------------
    if write_combined_csv:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(combined_path, index=False)

    print("latest date in the data is:", df["date"].max())
    print("total unique players:", df["player"].nunique())
    print("total gamelog rows:", len(df))
    return df
