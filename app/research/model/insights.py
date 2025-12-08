from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from app.constants import PLOTS_DIR, RANDOM_STATE
from app.research.model.evaluation import get_regression_metrics


def build_test_eval_frame(
    df_long: pd.DataFrame,
    mask_test: pd.Series,
    y_test: pd.Series,
    pipe: Pipeline,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Build a DataFrame for the TEST set.

    Contains:
    - utc_timestamp, country, is_weekend, is_holiday
    - y_true, y_pred, error, abs_error
    """
    y_pred = pipe.predict(X_test)

    # Align with original rows
    df_eval = df_long.loc[
        mask_test, ["utc_timestamp", "country", "is_weekend", "is_holiday"]
    ].copy()

    # Ensure alignment
    df_eval["y_true"] = y_test.values
    df_eval["y_pred"] = y_pred
    df_eval["error"] = df_eval["y_pred"] - df_eval["y_true"]
    df_eval["abs_error"] = df_eval["error"].abs()

    return df_eval.sort_values(["country", "utc_timestamp"]).reset_index(drop=True)


def per_country_metrics(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Compute regression metrics per country on the test set."""
    rows = []
    for country, g in df_eval.groupby("country"):
        metrics = get_regression_metrics(g["y_true"], g["y_pred"])
        metrics["country"] = country
        rows.append(metrics)

    df_country = pd.DataFrame(rows).set_index("country")
    df_country = df_country.sort_values("nrmse")

    print("\nPer-country metrics on TEST set (sorted by nRMSE):")
    print(df_country.round(2))

    return df_country


def permutation_importance_by_feature(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Estimate feature importance via permutation."""
    rng = np.random.RandomState(RANDOM_STATE)

    # Baseline performance
    y_pred_base = pipe.predict(X)
    baseline_rmse = np.sqrt(mean_squared_error(y, y_pred_base))

    rows: list[dict] = []

    for col in feature_cols:
        rmse_shuffled = []

        for _ in range(n_repeats):
            X_shuffled = X.copy()

            values = X_shuffled[col].to_numpy().copy()
            rng.shuffle(values)
            X_shuffled[col] = values

            y_pred_sh = pipe.predict(X_shuffled)
            rmse_sh = np.sqrt(mean_squared_error(y, y_pred_sh))
            rmse_shuffled.append(rmse_sh)

        mean_rmse_shuffled = float(np.mean(rmse_shuffled))
        delta_rmse = mean_rmse_shuffled - baseline_rmse

        rows.append(
            {
                "feature": col,
                "baseline_rmse": baseline_rmse,
                "shuffled_rmse": mean_rmse_shuffled,
                "delta_rmse": delta_rmse,
            }
        )

    df_imp = pd.DataFrame(rows).sort_values("delta_rmse", ascending=False)

    print("\nPermutation importance on TEST set (increase in RMSE when shuffled):")
    print(df_imp[["feature", "delta_rmse"]].round(4))

    return df_imp


def hourly_error_profile(
    df_eval: pd.DataFrame,
    output_dir: Path = PLOTS_DIR,
) -> pd.DataFrame:
    """Plot hourly error profile (hour vs mean_abs_error)."""
    print("\nPlotting hourly error profile:")

    df_hour = df_eval.copy()
    df_hour["hour"] = df_hour["utc_timestamp"].dt.hour

    agg = (
        df_hour.groupby("hour")["abs_error"]
        .mean()
        .reset_index()
        .rename(columns={"abs_error": "mean_abs_error"})
        .sort_values("hour")
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(agg["hour"], agg["mean_abs_error"], marker="o")
    plt.xticks(range(24))
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Mean absolute error [MW]")
    plt.title("Mean absolute error by hour-of-day (TEST set)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "hourly_error_profile.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved hourly error profile plot to {output_path}")

    return agg


def get_trained_model_insights(
    df_long: pd.DataFrame,
    mask_test: pd.Series,
    y_test: pd.Series,
    best_pipe: Pipeline,
    X_test: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    df_eval_test = build_test_eval_frame(
        df_long=df_long,
        mask_test=mask_test,
        y_test=y_test,
        pipe=best_pipe,
        X_test=X_test,
    )

    per_country_metrics(df_eval_test)
    permutation_importance_by_feature(
        best_pipe,
        X_test,
        y_test,
        feature_cols=feature_cols,
        n_repeats=5,
    )

    hourly_error_profile(df_eval_test)
