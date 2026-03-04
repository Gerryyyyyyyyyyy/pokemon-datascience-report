# pokemon-datascience-report

A reproducible, code-only data pipeline and analysis project on Pokémon base stats using the PokéAPI (Gen 1).

## Project structure

* `scripts/` - runnable pipeline steps (fetch, clean, eda, stats, hypothesis tests, ML regression, ML classification)
* `src/pokemon_ds/` - reusable code (fetching, cleaning, plotting, statistics, hypothesis testing, regression modeling, classification modeling)
* `data/` - raw/processed datasets (ignored by git)
* `reports/` - generated figures + `report.md` + JSON result files

## Setup

Create a virtual environment and install dependencies.

### Windows (CMD)

```
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -U pip
pip install pandas numpy matplotlib scipy requests pytest ruff black scikit-learn
```

### Windows (PowerShell)

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install pandas numpy matplotlib scipy requests pytest ruff black scikit-learn
```

## Current implementation (what is already built)

### 1) Data ingestion (raw data)

* Fetches Pokémon data from PokéAPI (`/pokemon/{id}`)
* Current scope: Gen 1 (first 151 Pokémon)
* Saves raw API data to:

  * `data/raw/pokemon_raw.csv`

### 2) Cleaning + feature engineering

* Cleans numeric columns
* Validates basic stat ranges
* Creates `total_stats` (sum of the six base stats)
* Saves cleaned dataset to:

  * `data/processed/pokemon_clean.csv`

### 3) EDA (visual analysis)

Implemented EDA plots and a markdown report (`reports/report.md`), including:

* distribution of `total_stats`
* type occurrences (all type appearances, including dual-types)
* primary types only
* top 10 Pokémon by `total_stats`

### 4) QA checks + summary statistics 

Implemented a reproducible statistics pipeline (`scripts/04_stats.py`) with:

* QA checks (row count, missing values, duplicate checks, `total_stats` consistency)
* mono-type vs dual-type counts
* global summary metrics
* primary type summary table (`count`, mean/median stats, speed)

Outputs:

* `reports/stats_results.json`
* `data/processed/primary_type_summary.csv`

### 5) Hypothesis testing 

Implemented a first hypothesis test:

**Question:** Do dual-type Pokémon have higher `total_stats` than mono-type Pokémon?

Methods included:

* group comparison (mono vs dual)
* bootstrap for **mean difference** (`dual - mono`)
* bootstrap for **median difference** (`dual - mono`) as a robustness check

Outputs:

* `reports/hypothesis_results.json`
* hypothesis plots in `reports/figures/`

### 6) Machine Learning (Regression + Cross-Validation)

Implemented a regression workflow to predict `base_experience` from base stats.

**Task:** Predict `base_experience` using:

* `hp`
* `attack`
* `defense`
* `special-attack`
* `special-defense`
* `speed`

Models compared:

* Baseline mean predictor
* Linear Regression
* Random Forest Regressor

ML workflow includes:

* train/test split (holdout evaluation)
* preprocessing with sklearn Pipelines (imputation + scaling where appropriate)
* cross-validation (5-fold) for more robust model comparison
* evaluation metrics: MAE, RMSE, R²
* Random Forest feature importance
* prediction-vs-actual diagnostic plots

Outputs:

* `reports/ml_regression_results.json`
* `data/processed/rf_feature_importance.csv`
* ML regression plots in `reports/figures/`

### 7) Machine Learning (Classification + Cross-Validation)

Implemented a binary classification workflow to predict whether a Pokémon is mono-type or dual-type from base stats.

**Task:** Predict `is_dual_type` using:

* `hp`
* `attack`
* `defense`
* `special-attack`
* `special-defense`
* `speed`

Models compared:

* Baseline majority-class predictor
* Logistic Regression
* Random Forest Classifier

ML workflow includes:

* target creation (`is_dual_type`) from `types`
* stratified train/test split (holdout evaluation)
* preprocessing with sklearn Pipelines (imputation + scaling where appropriate)
* cross-validation (5-fold) for robust model comparison
* evaluation metrics: Accuracy, Precision, Recall, F1
* confusion matrix component analysis
* Random Forest feature importance

Outputs:

* `reports/ml_classification_results.json`
* `data/processed/rf_classifier_feature_importance.csv`
* ML classification plots in `reports/figures/`

## How to run the current pipeline

> In PowerShell/CMD, set `PYTHONPATH=src` so imports from `src/pokemon_ds` work.

### PowerShell

```
$env:PYTHONPATH="src"

python .\scripts\01_fetch_raw_data.py
python .\scripts\02_clean.py
python .\scripts\03_eda.py
python .\scripts\04_stats.py
python .\scripts\05_hypothesis_tests.py
python .\scripts\06_train_regression.py
python .\scripts\07_train_classifier.py
```

### CMD

```
set PYTHONPATH=src

python .\scripts\01_fetch_raw_data.py
python .\scripts\02_clean.py
python .\scripts\03_eda.py
python .\scripts\04_stats.py
python .\scripts\05_hypothesis_tests.py
python .\scripts\06_train_regression.py
python .\scripts\07_train_classifier.py
```

## Current outputs (generated locally)

Examples of generated artifacts:

* `data/raw/pokemon_raw.csv`
* `data/processed/pokemon_clean.csv`
* `data/processed/primary_type_summary.csv`
* `data/processed/rf_feature_importance.csv`
* `data/processed/rf_classifier_feature_importance.csv`
* `reports/report.md`
* `reports/stats_results.json`
* `reports/hypothesis_results.json`
* `reports/ml_regression_results.json`
* `reports/ml_classification_results.json`
* plots in `reports/figures/`

## Notes

* Data source: PokéAPI (`https://pokeapi.co/`)
* Workflow style: code-only in VS Code (no notebooks)
* Generated data and figures are ignored by git via `.gitignore`
* This project is designed as a learning-focused end-to-end Data Science workflow (EDA → statistics → hypothesis testing → ML regression → ML classification)
