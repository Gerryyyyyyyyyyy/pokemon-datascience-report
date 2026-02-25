from pokemon_ds.config import RAW_DIR
from pokemon_ds.fetch import fetch_range

def main():

    # Create the raw data directory, fetch the data for the first 151 Pokemon and save it to a CSV file
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_range(limit=151)
    out = RAW_DIR / "pokemon_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} rows={len(df)}")

if __name__ == "__main__":
    main()