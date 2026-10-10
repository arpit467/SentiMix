import pandas as pd
import ast

for name, path in [
    ("Laptop Train", "data/complete_laptop_train.csv"),
    ("Laptop Test", "data/complete_laptop_test.csv"),
    ("Restaurant Train", "data/Complete_restaurant_train.csv"),
    ("Restaurant Test", "data/Complete_restaurant_test.csv")
]:
    df = pd.read_csv(path)
    print(f"\n=== {name} ({path}) ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    for i, r in df.head(5).iterrows():
        print(f"Sentence: {r['sentence']}")
        print(f"Aspect Terms: {r['aspect_terms']}")
        print(f"Polarity: {r['at_polarity']}")
        print("-")
