import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from src.preprocessor import TextPreprocessor
from src.models.rule_based import RuleBasedABSAModel

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class TFIDFLogisticABSAModel:
    """
    Model 2: Classical ML (TF-IDF + Multinomial Logistic Regression).
    Enhanced with hybrid lexicon prior blending for code-mixed robustness.
    """
    def __init__(self, c_reg: float = 1.0, max_iter: int = 1000):
        self.name = "TF-IDF + Logistic Regression"
        self.preprocessor = TextPreprocessor()
        self.rule_engine = RuleBasedABSAModel()
        self.c_reg = c_reg
        self.max_iter = max_iter
        
        # Word and Character TF-IDF Feature Union
        self.vectorizer = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=15000,
                sublinear_tf=True
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=25000,
                sublinear_tf=True
            ))
        ])
        
        self.classifier = LogisticRegression(
            C=self.c_reg,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        self.is_trained = False

    def _prepare_text_features(self, sentences: List[str], aspects: List[str]) -> List[str]:
        combined = []
        for s, a in zip(sentences, aspects):
            s_clean = self.preprocessor.normalize_hinglish(str(s))
            a_clean = self.preprocessor.normalize_hinglish(str(a))
            paired_text = f"ASPECT_{a_clean} {a_clean} [SEP] {s_clean}"
            combined.append(paired_text)
        return combined

    def _extract_sents(self, df: pd.DataFrame) -> List[str]:
        if "clean_text" in df.columns:
            return df["clean_text"].astype(str).tolist()
        elif "text" in df.columns:
            return df["text"].astype(str).tolist()
        elif "clean_sentence" in df.columns:
            return df["clean_sentence"].astype(str).tolist()
        elif "sentence" in df.columns:
            return df["sentence"].astype(str).tolist()
        return [""] * len(df)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Dict[str, float]:
        print(f"Training {self.name} on {len(train_df)} samples...")
        sents = self._extract_sents(train_df)
        asps = train_df["aspect"].astype(str).tolist()
        X_texts = self._prepare_text_features(sents, asps)
        y_train = train_df["label"].values
        
        X_vec = self.vectorizer.fit_transform(X_texts)
        self.classifier.fit(X_vec, y_train)
        self.is_trained = True
        
        train_acc = float(np.mean(self.classifier.predict(X_vec) == y_train))
        metrics = {"train_accuracy": train_acc}
        
        if val_df is not None and len(val_df) > 0:
            val_preds, _ = self.predict_batch(val_df)
            val_acc = float(np.mean(val_preds == val_df["label"].values))
            metrics["val_accuracy"] = val_acc
            print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
        return metrics

    def predict_pair(self, sentence: str, aspect: str) -> Tuple[str, int, Dict[str, float]]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        text_feat = self._prepare_text_features([sentence], [aspect])
        X_vec = self.vectorizer.transform(text_feat)
        ml_probs = self.classifier.predict_proba(X_vec)[0]
        
        # Light blend with lexicon prior (preserving ML as dominant predictor)
        _, _, rule_probs_dict = self.rule_engine.predict_pair(sentence, aspect)
        rule_probs = np.array([rule_probs_dict["negative"], rule_probs_dict["neutral"], rule_probs_dict["positive"]])
        
        weight_ml = 0.85
        weight_rule = 0.15
            
        blended_probs = weight_ml * ml_probs + weight_rule * rule_probs
        pred_id = int(np.argmax(blended_probs))
        
        probs_dict = {
            "negative": float(blended_probs[0]),
            "neutral": float(blended_probs[1]),
            "positive": float(blended_probs[2])
        }
        return INV_LABEL_MAP[pred_id], pred_id, probs_dict


    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        y_pred = []
        y_probs = []
        sents = self._extract_sents(df)
        asps = df["aspect"].astype(str).tolist()
        for sent, asp in zip(sents, asps):
            lbl, lid, probs = self.predict_pair(sent, asp)
            y_pred.append(lid)
            y_probs.append([probs["negative"], probs["neutral"], probs["positive"]])
        return np.array(y_pred), np.array(y_probs)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "classifier": self.classifier, "is_trained": self.is_trained}, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.is_trained = data["is_trained"]
        print(f"Model loaded from {filepath}")
