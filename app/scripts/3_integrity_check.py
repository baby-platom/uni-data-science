import json
from typing import Any

import numpy as np
import pandas as pd

from app.constants import (
    COUNTRY_CODES,
    INTEGRITY_JSON_REPORT_PATH,
    UNITED_DATASET_CSV_PATH,
)
from app.utils import round_floats


def check_timestamp(df: pd.DataFrame) -> dict[str, Any]:
    """Validate utc_timestamp.

    - strictly hourly frequency
    - no duplicates
    - no gaps between min and max timestamp
    """

    sorted_utc_timestamps = df["utc_timestamp"].sort_values()

    min_ts = sorted_utc_timestamps.iloc[0]
    max_ts = sorted_utc_timestamps.iloc[-1]

    expected_range = pd.date_range(start=min_ts, end=max_ts, freq="h", tz="UTC")
    actual_range = sorted_utc_timestamps.unique()

    missing_hours = expected_range.difference(actual_range)
    extra_timestamps = pd.Index(actual_range).difference(expected_range)

    return {
        "no_duplicates": df["utc_timestamp"].is_unique,
        "min_ts": str(min_ts),
        "max_ts": str(max_ts),
        "n_expected_hours": len(expected_range),
        "n_actual_hours": len(actual_range),
        "missing_hours_n": len(missing_hours),
        "extra_timestamps_n": len(extra_timestamps),
    }


def _summarize_numeric_series(
    s: pd.Series,
    column_name: str,
    non_negative: bool = False,
    hard_min: float | None = None,
    hard_max: float | None = None,
    iqr_outlier: bool = True,
) -> dict[str, Any]:
    """Compute descriptive stats and flag potential outliers / invalid values."""
    result: dict[str, Any] = {}

    s_numeric = pd.to_numeric(s, errors="coerce")
    n = len(s_numeric)

    if n_missing := int(s_numeric.isna().sum()):
        result["n_missing"] = n_missing
        result["missing_pct"] = float(n_missing / n * 100.0) if n > 0 else np.nan

    desc = s_numeric.describe()

    invalid_mask = pd.Series(False, index=s_numeric.index)

    if non_negative:
        neg_mask = s_numeric < 0
        invalid_mask |= neg_mask
        if int(neg_mask.sum()):
            raise ValueError

    if hard_min is not None:
        below_min = s_numeric < hard_min
        invalid_mask |= below_min
        if int(below_min.sum()):
            raise ValueError

    if hard_max is not None:
        above_max = s_numeric > hard_max
        invalid_mask |= above_max
        if int(above_max.sum()):
            raise ValueError

    outlier_mask = pd.Series(False, index=s_numeric.index)
    iqr_bounds = None

    if iqr_outlier and s_numeric.notna().sum() >= 10:
        q1 = s_numeric.quantile(0.25)
        q3 = s_numeric.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_bounds = (lower, upper)
            outlier_mask = (s_numeric < lower) | (s_numeric > upper)
        else:
            iqr_bounds = (None, None)

    if (iqr_outliers_n := int(outlier_mask.sum())) > 0:
        result["iqr_bounds"] = iqr_bounds
        result["iqr_outliers_pct"] = float(iqr_outliers_n / n * 100.0)

    result.update(
        {
            "column": column_name,
            "n_rows": int(n),
            "min": float(desc["min"])
            if "min" in desc and pd.notna(desc["min"])
            else None,
            "max": float(desc["max"])
            if "max" in desc and pd.notna(desc["max"])
            else None,
            "mean": float(desc["mean"]),
            "std": float(desc["std"])
            if "std" in desc and pd.notna(desc["std"])
            else None,
        }
    )

    return result


def check_load_columns(df: pd.DataFrame, country_codes: list[str]) -> dict[str, Any]:
    """Chec load columns for each country.

    - Load actual & forecast: missingness, non-negative, distribution, outliers
    """
    results: dict[str, Any] = {}

    for code in country_codes:
        actual_col = f"{code}_load_actual_entsoe_transparency"
        forecast_col = f"{code}_load_forecast_entsoe_transparency"

        actual_report = _summarize_numeric_series(
            df[actual_col],
            actual_col,
            non_negative=True,
        )
        forecast_report = _summarize_numeric_series(
            df[forecast_col],
            forecast_col,
            non_negative=True,
        )

        results[code] = {
            "actual": actual_report,
            "forecast": forecast_report,
        }

    return results


def check_temperature_and_radiation(
    df: pd.DataFrame, country_codes: list[str]
) -> dict[str, Any]:
    """Validate temperature and solar radiation columns for each country."""
    results: dict[str, Any] = {}

    for code in country_codes:
        temp_col = f"{code}_temperature"
        dir_col = f"{code}_radiation_direct_horizontal"
        diff_col = f"{code}_radiation_diffuse_horizontal"

        country_result: dict[str, Any] = {}

        country_result["temperature"] = _summarize_numeric_series(
            df[temp_col],
            temp_col,
            hard_min=-40.0,
            hard_max=50.0,
        )

        country_result["radiation_direct"] = _summarize_numeric_series(
            df[dir_col],
            dir_col,
            non_negative=True,
            hard_min=0.0,
            hard_max=1500.0,  # very conservative physical upper bound
        )

        country_result["radiation_diffuse"] = _summarize_numeric_series(
            df[diff_col],
            diff_col,
            non_negative=True,
            hard_min=0.0,
            hard_max=1500.0,
        )

        results[code] = country_result

    return results


def run_data_integrity_checks(
    df: pd.DataFrame,
    country_codes: list[str],
) -> dict[str, Any]:
    """Run all integrity checks on the dataset."""
    results: dict[str, Any] = {}

    # 1) utc_timestamp coverage & uniqueness
    results["utc_timestamp"] = check_timestamp(df)

    # 2) loads (actual, forecast) per country
    results["loads"] = check_load_columns(df, country_codes)

    # 3) temperature & radiation
    results["temperature_and_radiation"] = check_temperature_and_radiation(
        df, country_codes
    )

    return results


if __name__ == "__main__":
    dataset_df = pd.read_csv(UNITED_DATASET_CSV_PATH)
    dataset_df["utc_timestamp"] = pd.to_datetime(
        dataset_df["utc_timestamp"],
        utc=True,
    )

    report = run_data_integrity_checks(dataset_df, COUNTRY_CODES)
    report = round_floats(report)

    with INTEGRITY_JSON_REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=4)
