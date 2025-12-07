import pandas as pd


def _handle_missing_values(df_long: pd.DataFrame) -> pd.DataFrame:
    numeric_series_cols = [
        "load_actual",
        "temperature",
        "rad_dir",
        "rad_diff",
        "load_forecast",
    ]

    for col in numeric_series_cols:
        df_long[col] = df_long.groupby("country")[col].transform(
            lambda s: s.interpolate().ffill().bfill()
        )

    # Drop rows where target is still missing (unlikely)
    count_to_drop = df_long["load_actual"].isna().sum()
    print("Rows with still missing 'load_actual' count:", count_to_drop)
    df_long = df_long.dropna(subset=["load_actual"]).reset_index(drop=True)

    # Drop rows with missing key numeric features (unlikely)
    key_numeric_features = ["temperature", "rad_dir", "rad_diff"]
    count_to_drop = df_long[key_numeric_features].isna().any(axis=1).sum()
    print("Rows with still missing key numeric features count:", count_to_drop)
    df_long = df_long.dropna(subset=key_numeric_features)

    # Drop rows where "load_forecast" is still missing (unlikely)
    count_to_drop = df_long["load_forecast"].isna().sum()
    print("Rows with still missing 'load_forecast' count:", count_to_drop)
    df_long = df_long.dropna(subset=["load_forecast"]).reset_index(drop=True)

    df_long["is_weekend"] = df_long["is_weekend"].astype(bool)
    df_long["is_holiday"] = df_long["is_holiday"].astype(bool)

    return df_long


def preprocess_raw_long_dataset(df_raw_long: pd.DataFrame) -> pd.DataFrame:
    return _handle_missing_values(df_raw_long)
