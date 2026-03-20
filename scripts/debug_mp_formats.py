from __future__ import annotations

from pathlib import Path
import pandas as pd

from model_training.config import GAMELOG_PARQUET_ROOT


def _list_parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def main() -> None:
    root = Path(GAMELOG_PARQUET_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"Missing parquet root: {root}")

    files = _list_parquet_files(root)
    if not files:
        raise RuntimeError(f"No parquet files found under {root}")

    rows = []

    for p in files[:300]:
        try:
            df = pd.read_parquet(str(p), columns=None)
        except Exception as e:
            rows.append({
                "file": str(p),
                "season": None,
                "n_rows": None,
                "mp_dtype": f"ERROR: {type(e).__name__}",
                "mp_sample_1": None,
                "mp_sample_2": None,
                "mp_sample_3": None,
            })
            continue

        season = None
        if "season" in df.columns and len(df) > 0:
            try:
                season = pd.to_numeric(df["season"], errors="coerce").dropna()
                season = int(season.iloc[0]) if not season.empty else None
            except Exception:
                season = None

        mp_dtype = str(df["mp"].dtype) if "mp" in df.columns else "MISSING"

        mp_vals = []
        if "mp" in df.columns:
            mp_vals = (
                df["mp"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(10)
                .tolist()
            )

        rows.append({
            "file": str(p),
            "season": season,
            "n_rows": len(df),
            "mp_dtype": mp_dtype,
            "mp_sample_1": mp_vals[0] if len(mp_vals) > 0 else None,
            "mp_sample_2": mp_vals[1] if len(mp_vals) > 1 else None,
            "mp_sample_3": mp_vals[2] if len(mp_vals) > 2 else None,
        })

    out = pd.DataFrame(rows)

    print("\n=== MP DTYPE BY SEASON ===")
    if "season" in out.columns:
        print(
            out.groupby(["season", "mp_dtype"], dropna=False)
            .size()
            .reset_index(name="n_files")
            .sort_values(["season", "n_files"], ascending=[True, False])
            .to_string(index=False)
        )

    print("\n=== SAMPLE FILES ===")
    print(
        out[["season", "mp_dtype", "mp_sample_1", "mp_sample_2", "mp_sample_3", "file"]]
        .head(50)
        .to_string(index=False)
    )

    save_path = Path("debug_mp_formats.csv")
    out.to_csv(save_path, index=False)
    print(f"\nSaved debug file -> {save_path}")


if __name__ == "__main__":
    main()