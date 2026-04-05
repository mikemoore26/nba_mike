from pathlib import Path
import pandas as pd


MODEL_DIR = Path("models/rebounds")


def summarize_threshold(path: Path, actual_col: str, pred_col: str) -> dict:
    df = pd.read_csv(path, low_memory=False)

    if actual_col not in df.columns:
        raise ValueError(f"{path.name} missing actual col: {actual_col}")
    if pred_col not in df.columns:
        raise ValueError(f"{path.name} missing pred col: {pred_col}")

    mean_pred = float(df[pred_col].mean())
    actual_rate = float(df[actual_col].mean())

    return {
        "file": path.name,
        "n": int(len(df)),
        "mean_pred": mean_pred,
        "actual_rate": actual_rate,
        "diff_pred_minus_actual": mean_pred - actual_rate,
    }


def main():
    files = [
        (
            MODEL_DIR / "reb_ge_8_validation.csv",
            "reb_ge_8_actual",
            "reb_ge_8_pred_prob",
            "REB >= 8",
        ),
        (
            MODEL_DIR / "reb_ge_10_validation.csv",
            "reb_ge_10_actual",
            "reb_ge_10_pred_prob",
            "REB >= 10",
        ),
        (
            MODEL_DIR / "reb_ge_12_validation.csv",
            "reb_ge_12_actual",
            "reb_ge_12_pred_prob",
            "REB >= 12",
        ),
    ]

    for path, actual_col, pred_col, label in files:
        if not path.exists():
            raise ValueError(f"Missing file: {path}")

        out = summarize_threshold(path, actual_col, pred_col)

        print(f"\n=== {label} ===")
        print(out)


if __name__ == "__main__":
    main()