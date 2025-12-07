from pathlib import Path

RANDOM_STATE = 42

DATA_DIR = Path("data")

TIME_SERIES_CSV_PATH = DATA_DIR / "time_series_60min_singleindex.csv"
WEATHER_CSV_PATH = DATA_DIR / "weather_data.csv"

UNITED_DATASET_CSV_PATH = DATA_DIR / "united_dataset.csv"
INTEGRITY_JSON_REPORT_PATH = DATA_DIR / "integrity_report.json"

SOURCE_DATASETS_DOWNLOAD_URL_MAPPING = {
    TIME_SERIES_CSV_PATH: "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv",
    WEATHER_CSV_PATH: "https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv",
}

COUNTRY_CODES = [
    "AT",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GB",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SE",
    "SI",
    "SK",
]
