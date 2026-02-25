import pandas as pd
from pokemon_ds.config import PROCESSED_DIR, FIG_DIR
from pokemon_ds.eda import (
    save_total_stats_hist,
    save_top_types_occurrences,
    save_top_primary_types,
    save_top_pokemon_by_total_stats
)

def main():

    # Read the cleaned data, create the plots for total stats distribution
    # top type occurrences, and top primary types, and save them to the output directory
    df = pd.read_csv(PROCESSED_DIR / "pokemon_clean.csv")
    save_total_stats_hist(df, FIG_DIR)
    save_top_types_occurrences(df, FIG_DIR)
    save_top_primary_types(df, FIG_DIR)
    save_top_pokemon_by_total_stats(df, FIG_DIR)
    print("Saved plots to", FIG_DIR)
    

if __name__ == "__main__":
    main()