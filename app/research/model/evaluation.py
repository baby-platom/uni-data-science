import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def get_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
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

    metrics = get_regression_metrics(y, y_pred)

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

    metrics = get_regression_metrics(y_true_valid, y_baseline_valid)

    print(f"\n{name} metrics (baseline: {baseline_col}):")
    print(f"  MAE   : {metrics['mae']:.2f} MW")
    print(f"  RMSE  : {metrics['rmse']:.2f} MW")
    print(f"  WAPE  : {metrics['wape']:.2f} %")
    print(f"  MAPE  : {metrics['mape']:.2f} %")
    print(f"  nRMSE : {metrics['nrmse']:.2f} % of mean load")
    print(f"  R²    : {metrics['r2']:.4f}")
