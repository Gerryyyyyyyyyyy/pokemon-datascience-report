# pokemon-datascience-report

A reproducible data pipeline and EDA report on Pokemon base stats using the PokéAPI.

## Project structure
- `scripts/` - runnable pipeline steps (fetch, clean, eda)
- `src/pokemon_ds/` - reusable code (fetching, cleaning, plotting)
- `data/` - raw/processed datasets (ignored by git)
- `reports/` - generated figures + `report.md`

## Setup
Create a virtual environment and install dependencies:

### Windows (CMD)
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -U pip
pip install pandas numpy matplotlib scipy requests pytest 