import os
import sys
import ast
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessor import TextPreprocessor

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

class ABSADatasetLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.preprocessor = TextPreprocessor()

    @staticmethod
    def safe_parse_list(val: Any) -> List[str]:
        """Safely parse python list strings or returns empty list."""
        if isinstance(val, list):
            return val
        if not isinstance(val, str) or not val.strip():
            return []
        try:
            parsed = ast.literal_eval(val.strip())
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
            return [str(parsed).strip()]
        except Exception:
            cleaned = re.sub(r"[\[\]\'\"]", "", val).strip()
            return [x.strip() for x in cleaned.split(",") if x.strip()]

    def load_dataset(self, file_name: str) -> pd.DataFrame:
        """Loads a single CSV dataset and standardizes column names."""
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Standardize columns
        if "sentence" not in df.columns and "comments" in df.columns:
            df["sentence"] = df["comments"]
        elif "sentence" not in df.columns and "review_text" in df.columns:
            df["sentence"] = df["review_text"]
        
        # Normalize aspect columns
        if "aspect_terms" not in df.columns:
            df["aspect_terms"] = "[]"
        if "at_polarity" not in df.columns:
            df["at_polarity"] = "[]"
            
        df["aspect_terms_list"] = df["aspect_terms"].apply(self.safe_parse_list)
        df["at_polarity_list"] = df["at_polarity"].apply(self.safe_parse_list)
        df["clean_sentence"] = df["sentence"].apply(self.preprocessor.normalize_hinglish)
        df["is_hinglish"] = df["sentence"].apply(self.preprocessor.is_hinglish)
        return df

    def create_aspect_pairs(self, df: pd.DataFrame, domain: str = "general") -> pd.DataFrame:
        """
        Flattens sentence-level ABSA data into (sentence, aspect_term, polarity, label) pairs.
        """
        pairs = []
        for idx, row in df.iterrows():
            sent = str(row["sentence"])
            clean_sent = str(row.get("clean_sentence", self.preprocessor.normalize_hinglish(sent)))
            aspects = row["aspect_terms_list"]
            polarities = row["at_polarity_list"]
            is_hing = bool(row.get("is_hinglish", False))
            
            # Match aspects with polarities
            for i in range(min(len(aspects), len(polarities))):
                asp = str(aspects[i]).strip()
                pol = str(polarities[i]).strip().lower()
                
                # Normalize polarities
                if pol in ["conflict", "none", "mixed"]:
                    pol = "neutral"
                if pol not in LABEL_TO_ID:
                    continue
                
                pairs.append({
                    "sentence": sent,
                    "clean_sentence": clean_sent,
                    "aspect": asp,
                    "polarity": pol,
                    "label": LABEL_TO_ID[pol],
                    "is_hinglish": is_hing,
                    "domain": domain
                })
                
        return pd.DataFrame(pairs)

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Loads and standardizes all benchmark & e-commerce datasets into paired DataFrames.
        """
        datasets = {}
        
        # 1. Laptop Train & Test
        laptop_train_df = self.load_dataset("complete_laptop_train.csv")
        laptop_test_df = self.load_dataset("complete_laptop_test.csv")
        datasets["laptop_train"] = self.create_aspect_pairs(laptop_train_df, domain="laptop")
        datasets["laptop_test"] = self.create_aspect_pairs(laptop_test_df, domain="laptop")
        
        # 2. Restaurant Train & Test
        rest_train_df = self.load_dataset("Complete_restaurant_train.csv")
        rest_test_df = self.load_dataset("Complete_restaurant_test.csv")
        datasets["restaurant_train"] = self.create_aspect_pairs(rest_train_df, domain="restaurant")
        datasets["restaurant_test"] = self.create_aspect_pairs(rest_test_df, domain="restaurant")
        
        # 3. Meesho Annotated E-commerce Dataset
        if os.path.exists(os.path.join(self.data_dir, "meesho_annotated_absa.csv")):
            meesho_df = self.load_dataset("meesho_annotated_absa.csv")
            meesho_pairs = self.create_aspect_pairs(meesho_df, domain="meesho_ecommerce")
            train_m, test_m = train_test_split(meesho_pairs, test_size=0.2, random_state=42, stratify=meesho_pairs["label"])
            datasets["meesho_train"] = train_m.reset_index(drop=True)
            datasets["meesho_test"] = test_m.reset_index(drop=True)
            
        # 4. Flipkart Annotated E-commerce Dataset
        if os.path.exists(os.path.join(self.data_dir, "flipkart_reviews_annotated_absa.csv")):
            flip_df = self.load_dataset("flipkart_reviews_annotated_absa.csv")
            flip_pairs = self.create_aspect_pairs(flip_df, domain="flipkart_ecommerce")
            train_f, test_f = train_test_split(flip_pairs, test_size=0.2, random_state=42, stratify=flip_pairs["label"])
            datasets["flipkart_train"] = train_f.reset_index(drop=True)
            datasets["flipkart_test"] = test_f.reset_index(drop=True)

        # 5. Amazon India Annotated E-commerce Dataset
        if os.path.exists(os.path.join(self.data_dir, "amazon_reviews_annotated_absa.csv")):
            amz_df = self.load_dataset("amazon_reviews_annotated_absa.csv")
            amz_pairs = self.create_aspect_pairs(amz_df, domain="amazon_ecommerce")
            train_z, test_z = train_test_split(amz_pairs, test_size=0.2, random_state=42, stratify=amz_pairs["label"])
            datasets["amazon_train"] = train_z.reset_index(drop=True)
            datasets["amazon_test"] = test_z.reset_index(drop=True)
            
        # 6. Augmented Flipkart/Amazon E-commerce Dataset
        if os.path.exists(os.path.join(self.data_dir, "augmented_ecommerce_hinglish_absa.csv")):
            aug_df = self.load_dataset("augmented_ecommerce_hinglish_absa.csv")
            aug_pairs = self.create_aspect_pairs(aug_df, domain="ecommerce_electronics")
            train_a, test_a = train_test_split(aug_pairs, test_size=0.2, random_state=42, stratify=aug_pairs["label"])
            datasets["augmented_train"] = train_a.reset_index(drop=True)
            datasets["augmented_test"] = test_a.reset_index(drop=True)
            
        # 7. Combined Master Train and Test
        train_dfs = [datasets[k] for k in datasets if "train" in k]
        test_dfs = [datasets[k] for k in datasets if "test" in k]
        
        datasets["combined_train"] = pd.concat(train_dfs, ignore_index=True)
        datasets["combined_test"] = pd.concat(test_dfs, ignore_index=True)
        
        print("=== Dataset Summary Loaded ===")
        for k, v in datasets.items():
            print(f"{k:20s}: {len(v):5d} aspect-sentiment pairs")
            
        return datasets

if __name__ == "__main__":
    loader = ABSADatasetLoader()
    dsets = loader.load_all_datasets()
    print("\nSample from Combined Train:")
    print(dsets["combined_train"].head(10))
