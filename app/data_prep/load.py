import pandas as pd

from app.constants import COUNTRY_CODES, UNITED_DATASET_CSV_PATH


# Convert wide -> long (one row per country per hour)
def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["utc_timestamp", "date", "is_weekend"]
    long_parts = []

    for code in COUNTRY_CODES:
        col_map = {
            f"{code}_load_actual_entsoe_transparency": "load_actual",
            f"{code}_load_forecast_entsoe_transparency": "load_forecast",
            f"{code}_temperature": "temperature",
            f"{code}_radiation_direct_horizontal": "rad_dir",
            f"{code}_radiation_diffuse_horizontal": "rad_diff",
            f"{code}_is_holiday": "is_holiday",
        }

        cols_c = base_cols + list(col_map.keys())

        df_c = df_wide[cols_c].copy()
        df_c = df_c.rename(columns=col_map)
        df_c["country"] = code
        long_parts.append(df_c)

    df_long = pd.concat(long_parts, ignore_index=True)

    # Parse dates
    df_long["utc_timestamp"] = pd.to_datetime(df_long["utc_timestamp"])
    df_long["date"] = pd.to_datetime(df_long["date"])

    # Sort for time-series operations
    return df_long.sort_values(["country", "utc_timestamp"]).reset_index(drop=True)


def load_raw_long_dataset() -> pd.DataFrame:
    df_wide = pd.read_csv(UNITED_DATASET_CSV_PATH)
    return _wide_to_long(df_wide)
