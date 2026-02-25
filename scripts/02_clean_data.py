import pandas as pd
from pokemon_ds.config import RAW_DIR, PROCESSED_DIR
from pokemon_ds.clean import clean

def main():

    # Create the processed data directory, read the raw data, clean it, and save the cleaned data to a new CSV file
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "pokemon_raw.csv"
    df = pd.read_csv(raw_path)
    out_df = clean(df)
    out_path = PROCESSED_DIR / "pokemon_clean.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} rows={len(out_df)}")

if __name__ == "__main__":
    main()