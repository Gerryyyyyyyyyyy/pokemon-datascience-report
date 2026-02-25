from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def save_total_stats_hist(df: pd.DataFrame, out_dir: Path) -> None:

    # Create the output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot the histogram of total_stats and save it to the output directory
    fig, ax = plt.subplots()
    ax.hist(df["total_stats"].dropna(), bins=20)
    ax.set_title("Distribution of total base stats")
    ax.set_xlabel("total_stats")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "total_stats_hist.png", dpi=200)

    # Close the figure to free up memory
    plt.close(fig)

def save_top_types_occurrences(df: pd.DataFrame, out_dir: Path, top_n: int = 10) -> None:
    # Counts the occurrences of each type, including dual types
    s = (
        df["types"]
        .fillna("unknown")
        .str.split("|", regex=False)
        .explode()
        .str.strip()
    )
    counts = s.value_counts().head(top_n)

    # Plot the bar chart of the top type occurrences and save it to the output directory
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_title(f"Top {top_n} Type Occurrences (incl. dual types)")
    ax.set_xlabel("type")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "top_types_occurrences.png", dpi=200)

    # Close the figure to free up memory
    plt.close(fig)

def save_top_primary_types(df: pd.DataFrame, out_dir: Path, top_n: int = 10) -> None:

    # Counts the occurrences of each primary type and plots a bar chart of the top primary types, saving it to the output directory
    primary = (
        df["types"]
        .fillna("unknown")
        .str.split("|", regex=False)
        .str[0]
        .str.strip()
    )
    counts = primary.value_counts().head(top_n)

    # Plot the bar chart of the top primary types and save it to the output directory
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_title(f"Top {top_n} Primary Types")
    ax.set_xlabel("primary type")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "top_primary_types.png", dpi=200)
    
    # Close the figure to free up memory
    plt.close(fig)

def save_top_pokemon_by_total_stats(df: pd.DataFrame, out_dir: Path, top_n: int = 10) -> None:

    # Get the top pokemon by total stats and plot a bar chart of their names and total stats, saving it to the output directory
    top_pokemon = df[["name", "total_stats"]].dropna().sort_values("total_stats", ascending=False).head(top_n)

    # Plot the bar chart of the top pokemon by total stats and save it to the output directory
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_pokemon["name"], top_pokemon["total_stats"])
    ax.set_title(f"Top {top_n} Pokemon by total stats")
    ax.set_xlabel("total_stats")
    ax.set_ylabel("pokemon")
    ax.invert_yaxis()  # highest at top
    fig.tight_layout()
    fig.savefig(out_dir / "top_pokemon_by_total_stats.png", dpi=200)

    # Close the figure to free up memory
    plt.close(fig)