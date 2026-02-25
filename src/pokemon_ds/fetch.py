import time
import requests
import pandas as pd
from .config import POKEAPI_BASE

STAT_KEYS = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']

# Fetches Data of a single Pokemon by its ID and returns its Stats
def fetch_one(pokemon_id: int) -> dict:

    # Fetch the Pokemon data from the PokeAPI
    r = requests.get(f"{POKEAPI_BASE}/pokemon/{pokemon_id}", timeout=30)
    
    # Check if the request was successful, if not raise an HTTPError
    r.raise_for_status()

    # Parse the JSON response
    p = r.json()

    # Extract the relevant stats and types from the response
    stats = {s["stat"]["name"]: s["base_stat"] for s in p["stats"]}
    types = [t["type"]["name"] for t in p["types"]]

    # Create a dictionary to store the relevant data for the Pokemon
    row = {
        "id": p["id"],
        "name": p["name"],
        "height": p["height"],
        "weight": p["weight"],
        "base_experience": p.get("base_experience"),
        "types": "|".join(types),
    }

    # Add stats to the dictionary
    for k in STAT_KEYS:
        row[k] = stats.get(k)
    return row

# Fetches Data of a range of Pokemon by their IDs and returns a DataFrame
def fetch_range(limit: int, sleep_s: float = 0.2) -> pd.DataFrame:
    rows = []
    for i in range(1, limit + 1):
        rows.append(fetch_one(i))
        time.sleep(sleep_s)
    return pd.DataFrame(rows)