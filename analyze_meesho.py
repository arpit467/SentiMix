import pandas as pd

df = pd.read_csv("data/Multilingual_Meesho_Review_Dataset.csv")
print("Total rows:", len(df))
print("\nCategory distribution:")
print(df['category'].value_counts())
print("\nRating distribution:")
print(df['ratings'].value_counts())
print("\nMapped sentiment (sentence-level) distribution:")
print(df['mapped_sentiment'].value_counts())
