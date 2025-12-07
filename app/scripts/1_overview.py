import re

import pandas as pd

from app.constants import TIME_SERIES_CSV_PATH, WEATHER_CSV_PATH


def get_basic_overview(df: pd.DataFrame, df_name: str, top_columns_n: int = 25) -> None:
    print(f"\n{df_name} overview:")
    df.info(max_cols=top_columns_n)

    print(f"\ncolumns overview (top {top_columns_n} included):")
    print(list(df.columns)[:top_columns_n])


def get_country_codes(df: pd.DataFrame) -> set[str]:
    pattern = re.compile(r"^[A-Z]{2}_")

    result = set()

    for column in df.columns:
        if pattern.match(column):
            country_code = column[:2]
            result.add(country_code)

    return result


def describe_utc_timestamp(df: pd.DataFrame, df_name: str) -> None:
    print("\nDescribe `utc_timestamp`:")
    print(f"'{df_name}'")
    print(df["utc_timestamp"].describe())


def validate_utc_timestamp(df: pd.DataFrame, df_name: str) -> None:
    print("\nValidate `utc_timestamp`:")
    print(f"'{df_name}'")

    print(f"\nNo duplicates: {df['utc_timestamp'].is_unique}")

    sorted_utc_timestamps = df["utc_timestamp"].sort_values()

    min_ts = sorted_utc_timestamps.iloc[0]
    max_ts = sorted_utc_timestamps.iloc[-1]

    expected_range = pd.date_range(start=min_ts, end=max_ts, freq="h", tz="UTC")
    actual_range = sorted_utc_timestamps.unique()

    missing_hours = expected_range.difference(actual_range)
    extra_timestamps = pd.Index(actual_range).difference(expected_range)

    print("\nNumber of expected hours:", len(expected_range))
    print("Number of actual timestamps:", len(actual_range))
    print("Missing hours:", list(missing_hours))
    print("Unexpected (extra) timestamps:", list(extra_timestamps))


if __name__ == "__main__":
    time_series_df = pd.read_csv(TIME_SERIES_CSV_PATH)
    weather_df = pd.read_csv(WEATHER_CSV_PATH)

    # 0. Basic
    print("\n---Basic---\n")
    get_basic_overview(time_series_df, TIME_SERIES_CSV_PATH.name)
    get_basic_overview(weather_df, WEATHER_CSV_PATH.name)

    # 1. Check country codes
    print("\n---Country codes---\n")
    time_series_country_codes = get_country_codes(time_series_df)
    weather_country_codes = get_country_codes(weather_df)

    print("Country codes N")
    print(f"'{TIME_SERIES_CSV_PATH.name}': {len(time_series_country_codes)}")
    print(f"'{WEATHER_CSV_PATH.name}': {len(weather_country_codes)}")

    intersecting_country_codes = time_series_country_codes & weather_country_codes
    print(f"\nIntersecting country codes ({len(intersecting_country_codes)}):")
    print(sorted(intersecting_country_codes))

    # 2. Validate timestamps
    print("\n---Time stamps---\n")
    time_series_df["utc_timestamp"] = pd.to_datetime(
        time_series_df["utc_timestamp"],
        utc=True,
    )
    weather_df["utc_timestamp"] = pd.to_datetime(
        weather_df["utc_timestamp"],
        utc=True,
    )

    describe_utc_timestamp(time_series_df, TIME_SERIES_CSV_PATH.name)
    describe_utc_timestamp(weather_df, WEATHER_CSV_PATH.name)

    validate_utc_timestamp(time_series_df, TIME_SERIES_CSV_PATH.name)
    validate_utc_timestamp(weather_df, WEATHER_CSV_PATH.name)
