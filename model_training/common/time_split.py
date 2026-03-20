def time_split(
    df,
    *,
    split_date: str,
    date_col: str = "game_date",
):
    import pandas as pd

    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # Drop bad dates
    bad_dates = out[date_col].isna().sum()
    if bad_dates:
        print(f"[WARN] Dropping {bad_dates} rows with invalid {date_col}")
        out = out.dropna(subset=[date_col])

    if out.empty:
        raise ValueError("All rows dropped after date parsing")

    split_ts = pd.Timestamp(split_date)

    min_date = out[date_col].min()
    max_date = out[date_col].max()

    print(f"[INFO] Date range: {min_date.date()} → {max_date.date()}")
    print(f"[INFO] Split date: {split_ts.date()}")

    train = out[out[date_col] < split_ts].copy()
    test = out[out[date_col] >= split_ts].copy()

    print(f"[INFO] Train rows: {len(train)}")
    print(f"[INFO] Test rows: {len(test)}")

    if train.empty:
        raise ValueError(
            f"Train split empty. Available range: {min_date.date()} → {max_date.date()}"
        )
    if test.empty:
        raise ValueError(
            f"Test split empty. Available range: {min_date.date()} → {max_date.date()}"
        )

    return train, test