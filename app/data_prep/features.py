import numpy as np
import pandas as pd


# Time-based feature engineering
def _add_time_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    df["hour"] = df["utc_timestamp"].dt.hour
    df["dayofweek"] = df["utc_timestamp"].dt.dayofweek  # categorical
    df["dayofyear"] = df["utc_timestamp"].dt.dayofyear

    # Cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    return df


# Lag features (one-day lag: 24 hours)
def _add_lag_features(
    df_long: pd.DataFrame, lags: tuple[int, ...] = (24,)
) -> pd.DataFrame:
    df = df_long.sort_values(["country", "utc_timestamp"]).copy()
    for lag in lags:
        df[f"load_lag_{lag}"] = df.groupby("country")["load_actual"].shift(lag)

    # Drop rows with missing lag values (early in the series)
    lag_cols = [f"load_lag_{lag}" for lag in lags]
    return df.dropna(subset=lag_cols).reset_index(drop=True)


def add_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df_long = _add_time_features(df_long)
    return _add_lag_features(df_long)
