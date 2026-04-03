# Cell2Fire Static Data

This folder contains static lookup files required by `cell2fire_spread.py`.

## Required File

| File | Source |
|------|--------|
| `spain_lookup_table.csv` | Scott & Burgan 40 fuel model lookup — bundled with Cell2Fire |

## How to obtain

1. Clone or download the Cell2Fire repo:
   ```
   git clone https://github.com/cell2fire/Cell2Fire.git
   ```

2. Copy the lookup table into this folder:
   ```
   cp Cell2Fire/data/ScottAndBurgan/Hom_Fuel_101_40x40-asc/spain_lookup_table.csv \
      model-pipeline/src/models/obj2_spread/data/spain_lookup_table.csv
   ```

3. The simulator (`cell2fire_spread.py`) will automatically find it here — no
   environment variables or absolute paths needed.

## Why not committed?

The CSV is part of the Cell2Fire distribution and is not redistributed in this
repo to keep the repository size small. Every team member must copy it once
after cloning.
