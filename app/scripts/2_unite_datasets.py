from pathlib import Path

import holidays
import pandas as pd

from app.constants import (
    COUNTRY_CODES,
    TIME_SERIES_CSV_PATH,
    UNITED_DATASET_CSV_PATH,
    WEATHER_CSV_PATH,
)

TS_REQUIRED_SUFFIXES = [
    "load_actual_entsoe_transparency",
    "load_forecast_entsoe_transparency",
]

WEATHER_REQUIRED_SUFFIXES = [
    "temperature",
    "radiation_direct_horizontal",
    "radiation_diffuse_horizontal",
]

# GB uses "UK" in the holidays library
HOLIDAYS_CODE_BY_COUNTRY_MAPPING = {
    country: ("UK" if country == "GB" else country) for country in COUNTRY_CODES
}


# 1. Load CSVs


def load_raw_data(
    ts_path: Path, weather_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.read_csv(ts_path)
    weather = pd.read_csv(weather_path)

    ts["utc_timestamp"] = pd.to_datetime(ts["utc_timestamp"], utc=True)
    weather["utc_timestamp"] = pd.to_datetime(weather["utc_timestamp"], utc=True)

    return ts, weather


# 2. Normalize GB_GBN -> GB in the time series DF


def normalize_gb_in_time_series(ts: pd.DataFrame) -> pd.DataFrame:
    """Rename the 'GB_GBN_*' columns to 'GB_' in the time series data frame.

    Since in the weathre CSV there is only data for the 'GB', we omit the 'GB_NIR_*'
    (Northern Ireland) and 'GB_UKM_*' (United Kingdom) columns. For the normalization,
    we will treat 'GB_GBN_*' columns as just 'GB_*' ones.
    """
    ts = ts.copy()

    for suffix in TS_REQUIRED_SUFFIXES:
        old_col, new_col = f"GB_GBN_{suffix}", f"GB_{suffix}"

        if old_col not in ts.columns:
            raise ValueError("Missing required column")

        ts[new_col] = ts[old_col]

    return ts


# 3. Select time-series columns (one row per hour)


def select_time_series_columns(ts: pd.DataFrame) -> pd.DataFrame:
    """Select the columns of interest from the time series data frame.

    Keep:
    - utc_timestamp
    - {country}_{suffix} for all countries in `COUNTRY_CODES` and suffixes in
        `TS_REQUIRED_SUFFIXES`
    """
    ts_sel = pd.DataFrame()
    ts_sel["utc_timestamp"] = ts["utc_timestamp"]

    for country in COUNTRY_CODES:
        for suffix in TS_REQUIRED_SUFFIXES:
            col = f"{country}_{suffix}"
            if col not in ts.columns:
                raise ValueError("Missing required column")
            ts_sel[col] = ts[col]

    return ts_sel


# 4. Select weather columns


def select_weather_columns(weather: pd.DataFrame) -> pd.DataFrame:
    """Select the columns of interest from the weather data frame.

    Keep:
    - utc_timestamp
    - {country}_{suffix} for all countries in `COUNTRY_CODES` and suffixes in
        `WEATHER_REQUIRED_SUFFIXES`
    """
    weather_sel = pd.DataFrame()
    weather_sel["utc_timestamp"] = weather["utc_timestamp"]

    for country in COUNTRY_CODES:
        for suffix in WEATHER_REQUIRED_SUFFIXES:
            col = f"{country}_{suffix}"
            if col in weather.columns:
                weather_sel[col] = weather[col]

    return weather_sel


# 5. Merge time series and weather DFs


def merge_power_and_weather(
    ts_sel: pd.DataFrame,
    weather_sel: pd.DataFrame,
) -> pd.DataFrame:
    """Merge on utc_timestamp, keeping the intersecting time range."""

    return ts_sel.merge(
        weather_sel,
        on="utc_timestamp",
        how="inner",
        validate="one_to_one",
    )


# 6. Add weekend + per-country holiday flags


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features

    Features:
    - date (UTC date)
    - is_weekend (`True` if Saturday/Sunday)
    - {country}_is_holiday for each country in `COUNTRY_CODES`
    """
    df = df.copy()

    df["date"] = df["utc_timestamp"].dt.date
    df["is_weekend"] = df["utc_timestamp"].dt.weekday >= 5

    years = sorted(df["utc_timestamp"].dt.year.unique())

    for country in COUNTRY_CODES:
        holidays_code = HOLIDAYS_CODE_BY_COUNTRY_MAPPING[country]
        country_holidays = holidays.country_holidays(holidays_code, years=years)
        holiday_dates = set(country_holidays.keys())

        flag_col = f"{country}_is_holiday"
        df[flag_col] = df["date"].isin(holiday_dates)

    return df


# 7. Column ordering


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns.

    Put the global columns in the beginning and group the per-country ones.
    """

    ordered_cols = ["utc_timestamp", "date", "is_weekend"]
    used = set(ordered_cols)

    # Per-country blocks
    for country in COUNTRY_CODES:
        # 3 load-related
        for suffix in TS_REQUIRED_SUFFIXES:
            col = f"{country}_{suffix}"
            if col in df.columns and col not in used:
                ordered_cols.append(col)
                used.add(col)

        # 3 weather-related
        for suffix in WEATHER_REQUIRED_SUFFIXES:
            col = f"{country}_{suffix}"
            if col in df.columns and col not in used:
                ordered_cols.append(col)
                used.add(col)

        # Any other columns for this country
        prefix = f"{country}_"
        other_cols = [
            col for col in df.columns if col.startswith(prefix) and col not in used
        ]
        other_cols.sort()
        ordered_cols.extend(other_cols)
        used.update(other_cols)

    return df[ordered_cols]


# -------------------------------------------------------------------
# Full pipeline
# -------------------------------------------------------------------


def build_united_dataset(
    ts_path: Path = TIME_SERIES_CSV_PATH,
    weather_path: Path = WEATHER_CSV_PATH,
) -> pd.DataFrame:
    ts_raw, weather_raw = load_raw_data(ts_path, weather_path)

    ts_norm = normalize_gb_in_time_series(ts_raw)
    ts_sel = select_time_series_columns(ts_norm)
    weather_sel = select_weather_columns(weather_raw)

    merged = merge_power_and_weather(ts_sel, weather_sel)
    merged = add_calendar_features(merged)
    return order_columns(merged)


if __name__ == "__main__":
    df_final = build_united_dataset()
    print(df_final.head())
    print("Shape:", df_final.shape)

    df_final.to_csv(UNITED_DATASET_CSV_PATH, index=False)
    print(f"Saved united dataset to {UNITED_DATASET_CSV_PATH}")
