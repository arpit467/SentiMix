import pandas as pd
import glob
import os

print("=== Scanning Data Files ===")
data_files = glob.glob("data/*.csv")
for f in data_files:
    try:
        df = pd.read_csv(f)
        print(f"\n--- File: {f} ---")
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        print("Sample head:")
        print(df.head(2))
    except Exception as e:
        print(f"Error reading {f}: {e}")
