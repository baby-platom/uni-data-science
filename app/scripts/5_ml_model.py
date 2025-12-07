import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.constants import RANDOM_STATE
from app.data_prep import (
    add_features,
    load_raw_long_dataset,
    preprocess_raw_long_dataset,
)


# 1. Build time-based split
def time_based_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    ts_sorted = df["utc_timestamp"].sort_values()
    t1 = ts_sorted.quantile(0.70)
    t2 = ts_sorted.quantile(0.85)

    mask_train = df["utc_timestamp"] < t1
    mask_valid = (df["utc_timestamp"] >= t1) & (df["utc_timestamp"] < t2)
    mask_test = df["utc_timestamp"] >= t2

    return mask_train, mask_valid, mask_test


# 2. preprocessing + ML model pipeline
def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    max_depth: int = 8,
    learning_rate: float = 0.05,
    max_iter: int = 300,
) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


# 3. Evaluation helpers
def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute a set of useful regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # WAPE: sum(|e|) / sum(|y|)
    denom = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / denom * 100.0 if denom > 0 else np.nan

    # MAPE: mean(|e / y|) over non-zero y_true
    nonzero_mask = y_true != 0
    mape = (
        (
            np.mean(
                np.abs(
                    (y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]
                )
            )
            * 100.0
        )
        if np.any(nonzero_mask)
        else np.nan
    )

    # nRMSE: RMSE / mean(y)
    mean_y = np.mean(y_true)
    nrmse = rmse / mean_y * 100.0 if mean_y > 0 else np.nan

    # R^2
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "mape": mape,
        "nrmse": nrmse,
        "r2": r2,
    }


def evaluate_model(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series, name: str = "set"
) -> None:
    y_pred = pipe.predict(X)

    metrics = _regression_metrics(y, y_pred)

    print(f"\n{name} metrics (model):")
    print(f"  MAE   : {metrics['mae']:.2f} MW")
    print(f"  RMSE  : {metrics['rmse']:.2f} MW")
    print(f"  WAPE  : {metrics['wape']:.2f} %")
    print(f"  MAPE  : {metrics['mape']:.2f} %")
    print(f"  nRMSE : {metrics['nrmse']:.2f} % of mean load")
    print(f"  R²    : {metrics['r2']:.4f}")


def evaluate_baseline(
    df: pd.DataFrame,
    y_true: pd.Series,
    mask: pd.Series,
    baseline_col: str,
    name: str,
) -> None:
    """Evaluate a simple baseline on the subset given by `mask`.

    Args::
        `baseline_col`: column in `df` to use as prediction (e.g., 'load_lag_24' or
        'load_forecast').
    """
    y_baseline = df.loc[mask, baseline_col]
    valid_mask = ~y_baseline.isna()

    y_true_valid = y_true[valid_mask]
    y_baseline_valid = y_baseline[valid_mask]

    metrics = _regression_metrics(y_true_valid, y_baseline_valid)

    print(f"\n{name} metrics (baseline: {baseline_col}):")
    print(f"  MAE   : {metrics['mae']:.2f} MW")
    print(f"  RMSE  : {metrics['rmse']:.2f} MW")
    print(f"  WAPE  : {metrics['wape']:.2f} %")
    print(f"  MAPE  : {metrics['mape']:.2f} %")
    print(f"  nRMSE : {metrics['nrmse']:.2f} % of mean load")
    print(f"  R²    : {metrics['r2']:.4f}")


def main() -> None:
    # Load and reshape data
    df_raw_long = load_raw_long_dataset()
    df_long = preprocess_raw_long_dataset(df_raw_long)
    df_long = add_features(df_long)

    # Define features and target
    target_col = "load_actual"

    numeric_features = [
        "temperature",
        "rad_dir",
        "rad_diff",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "load_lag_24",
    ]

    categorical_features = [
        "country",
        "is_weekend",
        "is_holiday",
        "dayofweek",
    ]

    feature_cols = numeric_features + categorical_features

    X: pd.DataFrame = df_long[feature_cols]
    y: pd.Series = df_long[target_col]

    # time-based split
    mask_train, mask_valid, mask_test = time_based_split(df_long)

    X_train, y_train = X[mask_train], y[mask_train]
    X_valid, y_valid = X[mask_valid], y[mask_valid]
    X_test, y_test = X[mask_test], y[mask_test]

    print(
        f"Train size: {len(X_train)},"
        f" valid size: {len(X_valid)},"
        f" test size: {len(X_test)}"
    )

    # Hyperparameter search using the validation set
    param_grid = [
        {"max_depth": 6, "learning_rate": 0.05, "max_iter": 200},
        {"max_depth": 8, "learning_rate": 0.05, "max_iter": 300},
        {"max_depth": 10, "learning_rate": 0.05, "max_iter": 400},
        {"max_depth": 8, "learning_rate": 0.03, "max_iter": 400},
    ]

    best_rmse = np.inf
    best_params: dict = None

    print("\nHyperparameter search (validation RMSE):")
    for params in param_grid:
        pipe = build_model_pipeline(numeric_features, categorical_features, **params)
        pipe.fit(X_train, y_train)
        y_val_pred = pipe.predict(X_valid)
        val_mse = mean_squared_error(y_valid, y_val_pred)
        val_rmse = np.sqrt(val_mse)
        print(f"  {params} -> val RMSE = {val_rmse:8.2f} MW")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = params

    print(
        f"\nBest params based on validation RMSE: {best_params},"
        f" RMSE={best_rmse:.2f} MW"
    )

    # Retrain best model on train + validation
    X_trainval = pd.concat([X_train, X_valid], axis=0)
    y_trainval = pd.concat([y_train, y_valid], axis=0)

    best_pipe = build_model_pipeline(
        numeric_features,
        categorical_features,
        **best_params,
    )
    best_pipe.fit(X_trainval, y_trainval)

    # Final evaluation on held-out test set
    print("\nFinal model performance on TEST set (unseen during tuning):")
    evaluate_model(best_pipe, X_test, y_test, name="test")

    # Baselines on test set
    print("\nBaselines on TEST set:")

    # Persistence baseline: "tomorrow's load = today's load"
    evaluate_baseline(
        df_long,
        y_test,
        mask_test,
        baseline_col="load_lag_24",
        name="lag24",
    )

    # Official ENTSO-E day-ahead forecast
    evaluate_baseline(
        df_long,
        y_test,
        mask_test,
        baseline_col="load_forecast",
        name="entsoe",
    )


if __name__ == "__main__":
    main()
