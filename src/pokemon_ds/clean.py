import pandas as pd

STAT_COLS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]

# Cleans the DataFrame 
def clean(df: pd.DataFrame) -> pd.DataFrame:
    to_be_cleaned = df.copy()

    # Fill missing values in "types" with "null"
    to_be_cleaned["types"] = to_be_cleaned["types"].fillna("null")

    # Convert numeric columns to numeric types, coercing errors to NaN
    num_cols = ["height", "weight", "base_experience"] + STAT_COLS
    for c in num_cols:
        to_be_cleaned[c] = pd.to_numeric(to_be_cleaned[c], errors="coerce")

    # Calculate total stats as the sum of individual stats
    to_be_cleaned["total_stats"] = to_be_cleaned[STAT_COLS].sum(axis=1)

    # stats have to be between 1 and 255
    for c in STAT_COLS:
        to_be_cleaned = to_be_cleaned[to_be_cleaned[c].between(1, 255, inclusive="both")]

    return to_be_cleaned.reset_index(drop=True)