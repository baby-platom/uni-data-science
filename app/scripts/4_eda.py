from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.constants import PLOTS_DIR
from app.data_prep import (
    add_features,
    load_raw_long_dataset,
    preprocess_raw_long_dataset,
)


def plot_avg_load_by_hour(
    df: pd.DataFrame,
    country_code: str | None = None,
    output_dir: Path = PLOTS_DIR,
) -> None:
    data = df[df["country"] == country_code].copy() if country_code is not None else df

    hourly = (
        data.groupby("hour")["load_actual"].mean().reset_index().sort_values("hour")
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(hourly["hour"], hourly["load_actual"], marker="o")
    plt.xticks(range(24))
    title = "Average Load by Hour of Day"
    if country_code:
        title += f" ({country_code})"
    plt.title(title)
    plt.xlabel("Hour of day (0-23, UTC)")
    plt.ylabel("Average load (MW)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    file_name = (
        f"avg_load_by_hour_{country_code}.png"
        if country_code
        else "avg_load_by_hour.png"
    )
    output_path = Path(output_dir) / file_name
    plt.savefig(output_path, dpi=150)

    print(f"Saved avg_load_by_hour plot to {output_path}")
    plt.close()


DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def plot_avg_load_by_weekday(
    df: pd.DataFrame,
    country_code: str | None = None,
    output_dir: Path = PLOTS_DIR,
) -> None:
    data = df[df["country"] == country_code].copy() if country_code is not None else df

    weekday = data.groupby("dayofweek")["load_actual"].mean().reset_index()
    weekday["day_name"] = weekday["dayofweek"].map(DAY_NAMES)
    weekday = weekday.sort_values("dayofweek")

    y = weekday["load_actual"]
    x = weekday["day_name"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    bars = plt.bar(x, y)

    y_min = y.min()
    y_max = y.max()
    margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
    plt.ylim(y_min - margin, y_max + margin)

    # Add value labels on top of each bar
    for bar, val in zip(bars, y, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    title = "Average Load by Day of Week"
    if country_code:
        title += f" ({country_code})"
    plt.title(title)
    plt.xlabel("Day of week")
    plt.ylabel("Average load (MW)")
    plt.tight_layout()

    file_name = (
        f"avg_load_by_weekday_{country_code}.png"
        if country_code
        else "avg_load_by_weekday.png"
    )
    output_path = Path(output_dir) / file_name
    plt.savefig(output_path, dpi=150)

    print(f"Saved avg_load_by_weekday plot to {output_path}")
    plt.close()


def plot_avg_load_by_temperature_bin(
    df: pd.DataFrame,
    country_code: str | None = None,
    output_dir: Path = PLOTS_DIR,
    bin_width: float = 2.0,
) -> None:
    data = df[df["country"] == country_code].copy() if country_code is not None else df

    temp_min = np.floor(data["temperature"].min())
    temp_max = np.ceil(data["temperature"].max())
    bins = np.arange(temp_min, temp_max + bin_width, bin_width)

    data = data.copy()
    data["temp_bin"] = pd.cut(data["temperature"], bins=bins)

    temp_stats = (
        data.groupby("temp_bin", observed=True)["load_actual"].mean().reset_index()
    )
    temp_stats["temp_mid"] = temp_stats["temp_bin"].apply(
        lambda x: x.left + (x.right - x.left) / 2
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(temp_stats["temp_mid"], temp_stats["load_actual"], marker="o")
    title = f"Average Load by Temperature (bin width = {bin_width}°C)"
    if country_code:
        title += f" ({country_code})"
    plt.title(title)
    plt.xlabel("Temperature (°C, bin midpoints)")
    plt.ylabel("Average load (MW)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    file_name = (
        f"avg_load_by_temperature_bin_{country_code}.png"
        if country_code
        else "avg_load_by_temperature_bin.png"
    )
    output_path = Path(output_dir) / file_name
    plt.savefig(output_path, dpi=150)

    print(f"Saved avg_load_by_temperature_bin plot to {output_path}")
    plt.close()


def main() -> None:
    df_raw_long = load_raw_long_dataset()
    df_long = preprocess_raw_long_dataset(df_raw_long)
    df_long = add_features(df_long)

    plot_avg_load_by_hour(df_long)
    plot_avg_load_by_hour(df_long, "DE")

    plot_avg_load_by_weekday(df_long)
    plot_avg_load_by_weekday(df_long, "DE")

    plot_avg_load_by_temperature_bin(df_long, "DE")


if __name__ == "__main__":
    main()
