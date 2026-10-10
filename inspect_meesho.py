import pandas as pd

df_meesho = pd.read_csv("data/Multilingual_Meesho_Review_Dataset.csv")
print("Meesho Shape:", df_meesho.shape)
print("Meesho Columns:", df_meesho.columns.tolist())
print("\nMissing comments:", df_meesho['comments'].isna().sum())
print("\nSample comments (first 15):")
for i, r in df_meesho.dropna(subset=['comments']).head(15).iterrows():
    print(f"[{r['category']} | Rating: {r.get('ratings', 'N/A')}] {r['comments']}")
