from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.constants import PLOTS_DIR


def _month_to_season(m: int) -> str:
    if m in (12, 1, 2):
        return "winter"
    if m in (6, 7, 8):
        return "summer"
    return "shoulder"


def rq_temperature_sensitivity(
    df_long: pd.DataFrame,
    country_code: str | None = None,
    output_dir: Path = PLOTS_DIR,
) -> pd.DataFrame:
    """Analyze the temperature sensitivity RQ.

    How does the relationship between hourly electricity load and temperature
    differ between seasons (winter, summer, shoulder) for a given region?

    If `country_code` is `None`, all countries together are considered.
    """
    df = df_long.copy()

    region_label = "ALL"
    if country_code is not None:
        df = df[df["country"] == country_code].copy()
        region_label = country_code

    df["month"] = df["utc_timestamp"].dt.month
    df["season"] = df["month"].apply(_month_to_season)

    rows = []
    for season, group in df.groupby("season"):
        min_group_size = 100
        if len(group) < min_group_size:
            continue

        temp = group["temperature"].to_numpy()
        load = group["load_actual"].to_numpy()

        if np.isnan(temp).all() or np.isnan(load).all():
            continue
        if np.nanstd(temp) == 0 or np.nanstd(load) == 0:
            continue

        corr = np.corrcoef(temp, load)[0, 1]

        try:
            slope, _ = np.polyfit(temp, load, deg=1)
        except np.linalg.LinAlgError:
            slope, _, corr = np.nan, np.nan, np.nan

        rows.append(
            {
                "region": region_label,
                "season": season,
                "n_obs": len(group),
                "corr_temp_load": corr,
                "slope_MW_per_degC": slope,
            }
        )

    result = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)

    print(f"\nTemperature-load relationship for region={region_label}:")
    print(result)

    plot_rq_temperature_sensitivity(
        result,
        region_label=region_label,
        output_dir=output_dir,
    )
    return result


def plot_rq_temperature_sensitivity(
    result: pd.DataFrame,
    region_label: str,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Bar chart of slope (MW/°C) by season for a region."""
    output_dir.mkdir(parents=True, exist_ok=True)

    seasons_order = ["winter", "shoulder", "summer"]
    result = result.set_index("season").reindex(seasons_order).reset_index()
    result = result.dropna(subset=["slope_MW_per_degC"])

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(result["season"], result["slope_MW_per_degC"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel("Slope of load vs temperature [MW / °C]")
    ax.set_title(f"Temperature sensitivity by season - {region_label}")
    fig.tight_layout()

    plot_path = output_dir / f"rq_temperature_sensitivity_{region_label}.png"
    fig.savefig(plot_path, dpi=150)

    print(f"Saved plot to {plot_path}")
    plt.close(fig)
