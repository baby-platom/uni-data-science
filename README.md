# uni-data-science
The repository contains the code of the End-to-End Data Science / Computer Science / Software Engineering project. The topic is "Machine Learning Forecasting and Analysis of European Electricity Markets".

## Table of Contents
1. [Overview](#overview)
3. [Installation](#installation)
3. [Datasets](#datasets)
4. [Usage](#usage)

## Overview
There are two root folders: `app` and `data`.
1. `app` contains the project's code
2. `data` keeps the used dataset.

## Installation
The required Python version is `3.12.7`

1. Clone the repo
```sh
git clone https://github.com/baby-platom/uni-data-science.git
```

2. Create virtual environment and install the dependencies
```sh
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Optionally: 
- Use [uv](https://docs.astral.sh/uv/) for dependencies management
- Use [ruff](https://docs.astral.sh/ruff/) as a linter and code formatter

## Datasets
The research relies on two public datasets from [Open Power System Data](https://open-power-system-data.org/). The following data packages are taken from there:
- `time_series_60min_singleindex.csv` - https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv
- `weather_data.csv` - https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv 

Since these CSV files are too big to keep in the Github repository, please download them and put in the `data/` folder before using them. In the `app/constants.py` file you can view what exact file paths are expected.

**Recommended**: Instead of downloading the CSV files by hand, just run the `app/scripts/0_download_data.py` which automatically downloads the data packages to the `data/` directory.

## Usage
The runnable scripts are all inside the `app/scripts/` directory. Each file has a number prefix, according to which the scripts are ordered. Run one script file after another in numeric order to complete every corresponding step of the research. You can run a script using the command of the `python -m app.scripts.0_download_data` format.

View the console output. The plots are saved to `data/plots/` directory.