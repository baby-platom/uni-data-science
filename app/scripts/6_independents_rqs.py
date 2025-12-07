from app.data_prep import (
    add_features,
    load_raw_long_dataset,
    preprocess_raw_long_dataset,
)
from app.research.independent_questions import rq_temperature_sensitivity


def main() -> None:
    df_raw_long = load_raw_long_dataset()
    df_long = preprocess_raw_long_dataset(df_raw_long)
    df_long = add_features(df_long)

    # Europe average temperature sensitivity
    rq_temperature_sensitivity(df_long)

    # Country-level temperature sensitivity
    country_codes = ["DE", "FR", "GB", "ES", "IT"]
    for code in country_codes:
        rq_temperature_sensitivity(df_long, code)


if __name__ == "__main__":
    main()
