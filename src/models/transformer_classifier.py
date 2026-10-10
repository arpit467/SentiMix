import os
import sys
import src.torch_fix
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig
from sklearn.linear_model import LogisticRegression
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocessor import TextPreprocessor

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class BaseTransformerABSAModel:
    """
    Base class for Transformer-based ABSA (Sentence-Pair: [CLS] sentence [SEP] aspect [SEP]).
    Extracts deep contextual embeddings and trains a calibrated classification head.
    """
    def __init__(self, model_id: str, display_name: str, max_length: int = 128, device: str = None):
        self.model_id = model_id
        self.name = display_name
        self.max_length = max_length
        self.preprocessor = TextPreprocessor()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing {self.name} ({self.model_id}) on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.transformer = AutoModel.from_pretrained(self.model_id)
        self.transformer.to(self.device)
        self.transformer.eval()
        
        # Calibrated classifier head on top of contextual embeddings
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        self.is_trained = False

    def _extract_embeddings(self, sentences: List[str], aspects: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        num_samples = len(sentences)
        
        for i in range(0, num_samples, batch_size):
            batch_sents = [self.preprocessor.normalize_hinglish(str(s)) for s in sentences[i:i + batch_size]]
            batch_asps = [self.preprocessor.normalize_hinglish(str(a)) for a in aspects[i:i + batch_size]]
            
            inputs = self.tokenizer(
                batch_sents,
                batch_asps,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.transformer(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_emb = outputs.pooler_output.cpu().numpy()
                else:
                    # Mean pooling over attention mask
                    last_hidden = outputs.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    batch_emb = (torch.sum(last_hidden * mask, 1) / sum_mask).cpu().numpy()
                    
            all_embeddings.append(batch_emb)
            
        return np.vstack(all_embeddings)

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

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, batch_size: int = 32) -> Dict[str, float]:
        print(f"\n--- Training {self.name} on {len(train_df)} samples ---")
        sents = self._extract_sents(train_df)
        asps = train_df["aspect"].astype(str).tolist()
        X_train_emb = self._extract_embeddings(sents, asps, batch_size=batch_size)
        y_train = train_df["label"].values
        
        self.classifier.fit(X_train_emb, y_train)
        self.is_trained = True
        
        train_preds = self.classifier.predict(X_train_emb)
        train_acc = float(np.mean(train_preds == y_train))
        metrics = {"train_accuracy": train_acc}
        
        if val_df is not None and len(val_df) > 0:
            val_preds, _ = self.predict_batch(val_df, batch_size=batch_size)
            val_acc = float(np.mean(val_preds == val_df["label"].values))
            metrics["val_accuracy"] = val_acc
            print(f"[{self.name}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        else:
            print(f"[{self.name}] Train Acc: {train_acc:.4f}")
            
        return metrics

    def predict_pair(self, sentence: str, aspect: str) -> Tuple[str, int, Dict[str, float]]:
        if not self.is_trained:
            raise ValueError(f"Model {self.name} is not trained yet.")
        emb = self._extract_embeddings([sentence], [aspect], batch_size=1)
        pred_id = int(self.classifier.predict(emb)[0])
        pred_probs = self.classifier.predict_proba(emb)[0]
        
        probs_dict = {
            "negative": float(pred_probs[0]),
            "neutral": float(pred_probs[1]),
            "positive": float(pred_probs[2])
        }
        return INV_LABEL_MAP[pred_id], pred_id, probs_dict

    def predict_batch(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise ValueError(f"Model {self.name} is not trained yet.")
        sents = self._extract_sents(df)
        asps = df["aspect"].astype(str).tolist()
        emb = self._extract_embeddings(sents, asps, batch_size=batch_size)
        y_pred = self.classifier.predict(emb)
        y_probs = self.classifier.predict_proba(emb)
        return y_pred, y_probs

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        head_path = os.path.join(dirpath, "classifier_head.joblib")
        joblib.dump({"classifier": self.classifier, "model_id": self.model_id, "name": self.name, "is_trained": self.is_trained}, head_path)
        print(f"Model checkpoint saved to {dirpath}")

    def load(self, dirpath: str):
        head_path = os.path.join(dirpath, "classifier_head.joblib")
        data = joblib.load(head_path)
        self.classifier = data["classifier"]
        self.is_trained = data["is_trained"]
        print(f"Model checkpoint loaded from {dirpath}")


class IndicBERTABSAModel(BaseTransformerABSAModel):
    """Model 3: IndicBERT (ai4bharat/IndicBERTv2-MLM-only)"""
    def __init__(self, model_id: str = "ai4bharat/IndicBERTv2-MLM-only", max_length: int = 128, device: str = None):
        super().__init__(model_id=model_id, display_name="IndicBERT", max_length=max_length, device=device)


class MultilingualMiniLMABSAModel(BaseTransformerABSAModel):
    """Model 4: Multilingual MiniLM (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) - Ultra Efficient"""
    def __init__(self, model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_length: int = 128, device: str = None):
        super().__init__(model_id=model_id, display_name="Multilingual MiniLM", max_length=max_length, device=device)


class XLMRoBERTaABSAModel(BaseTransformerABSAModel):
    """Model 5: XLM-RoBERTa (xlm-roberta-base)"""
    def __init__(self, model_id: str = "xlm-roberta-base", max_length: int = 128, device: str = None):
        super().__init__(model_id=model_id, display_name="XLM-RoBERTa", max_length=max_length, device=device)


class MuRILABSAModel(BaseTransformerABSAModel):
    """Model 6: MuRIL (google/muril-base-cased)"""
    def __init__(self, model_id: str = "google/muril-base-cased", max_length: int = 128, device: str = None):
        super().__init__(model_id=model_id, display_name="MuRIL", max_length=max_length, device=device)
