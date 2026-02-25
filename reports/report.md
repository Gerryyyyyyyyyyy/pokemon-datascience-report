# Pokemon Data Science Report (Gen 1)

## Data source
PokéAPI (`/pokemon/{id}`)

## Pipeline
Fetch raw CSV → clean CSV → EDA plots

## Current outputs
- `data/raw/pokemon_raw.csv`
- `data/processed/pokemon_clean.csv`

## Initial findings

### 1) Distribution of total stats
![Total stats distribution](figures/total_stats_hist.png)

- A total base stat value around **500** appears very frequently in this dataset.
- Many Pokemon cluster in a mid to high stat range rather than being uniformly distributed.
- There appear to be more lower-stat outliers than very high-stat outliers.

> Note: This interpretation should be validated by checking exact counts, since visual binning in histograms can influence how peaks appear.

### 2) Type occurrences (all types, including dual types)
![Type occurrences](figures/top_types_occurrences.png)

- **Poison** and **Water** are among the most frequent type occurrences in Gen 1.
- The distribution is uneven: a few types appear very often, while several types are relatively rare.

### 3) Primary types only
![Primary types](figures/top_primary_types.png)

- **Water** appears to be the most common primary type in Gen 1.
- Compared to total type occurrences, **Poison** drops significantly, which suggests it often appears as a secondary type.
- **Normal** appears frequently as a primary/mono type and seems less dependent on dual-typing than some other types.

### 4) Top 10 Pokemon by total stats
![Top Pokemon by total stats](figures\top_pokemon_by_total_stats.png)

- This plot highlights the highest-stat Pokemon in the Gen 1 dataset.
- 5 of the top 10 Pokemon are legendary.
- In addition 1 Pokemon is often referred to as a “pseudo-legendary”, meaning more than half of the top 10 are rare/high-tier Pokémon by common fan classification.
- This supports the expectation that the upper end of the `total_stats` distribution is dominated by legendary Pokemon.

## Notes
- Type occurrences count all type appearances (dual-type Pokemon count twice).
- Primary type counts only the first listed type per Pokemon.
- Differences between the two plots highlight how counting definitions affect interpretation.