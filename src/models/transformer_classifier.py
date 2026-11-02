import os
import sys
import src.torch_fix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


class ABSAPairDataset(Dataset):
    """PyTorch Dataset for sentence-aspect pairs."""
    def __init__(self, sentences: List[str], aspects: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.sentences = sentences
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = str(self.sentences[idx])
        asp = str(self.aspects[idx])
        label = int(self.labels[idx]) if self.labels is not None else 0

        encoding = self.tokenizer(
            sent,
            asp,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class BaseTransformerABSAModel:
    """
    Universal Transformer Model for ABSA:
    Supports BOTH:
    1. True End-to-End Fine-Tuning (AutoModelForSequenceClassification with AdamW backprop)
    2. Linear Probing / Feature Extraction (Frozen backbone + Logistic Regression head)
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
        
        # End-to-end fine-tuned sequence classification model instance
        self.fine_tuned_model: Optional[AutoModelForSequenceClassification] = None
        
        # Frozen feature extraction base model
        self.transformer: Optional[AutoModel] = None
        
        # Calibrated classifier head for linear probing
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        self.is_trained = False
        self.is_fine_tuned = False

    def _get_base_transformer(self) -> AutoModel:
        if self.transformer is None:
            self.transformer = AutoModel.from_pretrained(self.model_id)
            self.transformer.to(self.device)
            self.transformer.eval()
        return self.transformer

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

    def fine_tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        epochs: int = 2,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        max_samples: int = None,
        checkpoint_dir: Optional[str] = None,
        use_amp: bool = True
    ) -> Dict[str, Any]:
        """
        True End-to-End Fine-Tuning using PyTorch and HuggingFace AutoModelForSequenceClassification.
        Supports:
        - Automatic Mixed Precision (AMP / FP16) for 3x GPU acceleration
        - Automatic Per-Epoch Checkpointing to resume or protect training progress
        """
        use_cuda_amp = use_amp and (str(self.device).startswith("cuda"))
        print(f"\n{'='*60}")
        print(f"  END-TO-END FINE-TUNING: {self.name} ({self.model_id})")
        print(f"  Device: {self.device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | AMP (FP16): {use_cuda_amp}")
        print(f"{'='*60}")
        sys.stdout.flush()

        if max_samples is not None and len(train_df) > max_samples:
            train_data = train_df.sample(max_samples, random_state=42).reset_index(drop=True)
            print(f"Subsampled {max_samples} balanced records for fine-tuning...")
        else:
            train_data = train_df.reset_index(drop=True)

        sents_train = [self.preprocessor.normalize_hinglish(s) for s in self._extract_sents(train_data)]
        asps_train = [self.preprocessor.normalize_hinglish(a) for a in train_data["aspect"].astype(str).tolist()]
        labels_train = train_data["label"].astype(int).tolist()

        train_dataset = ABSAPairDataset(sents_train, asps_train, labels_train, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(self.device != "cpu"))

        # Initialize sequence classification model (or resume from checkpoint if available)
        resumed = False
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            try:
                print(f"Resuming fine-tuning from existing checkpoint at {checkpoint_dir}...")
                self.fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(self.device)
                resumed = True
            except Exception as e:
                print(f"Notice: Could not load existing checkpoint ({e}). Initializing fresh model...")

        if not resumed:
            self.fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                num_labels=3
            ).to(self.device)

        self.fine_tuned_model.train()

        # Class weights to handle imbalance
        counts = np.bincount(labels_train, minlength=3)
        total_samples = len(labels_train)
        weights = [total_samples / (3.0 * max(c, 1)) for c in counts]
        class_weights_t = torch.tensor(weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        optimizer = torch.optim.AdamW(self.fine_tuned_model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

        total_steps = len(train_loader) * epochs
        step = 0

        for epoch in range(epochs):
            self.fine_tuned_model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            sys.stdout.flush()
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    outputs = self.fine_tuned_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.fine_tuned_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                step += 1

                if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0 or (batch_idx + 1) == len(train_loader):
                    current_acc = (correct / total) * 100.0
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {avg_loss:.4f} - Accuracy: {current_acc:.2f}%")
                    sys.stdout.flush()

            # Automatic per-epoch checkpointing
            if checkpoint_dir:
                print(f"--> Saved checkpoint at end of Epoch {epoch + 1} to {checkpoint_dir}")
                self.save(checkpoint_dir)
                sys.stdout.flush()

        self.fine_tuned_model.eval()
        self.is_fine_tuned = True
        self.is_trained = True

        val_acc = 0.0
        if val_df is not None and len(val_df) > 0:
            val_preds, _ = self.predict_batch(val_df, batch_size=batch_size)
            val_acc = float(np.mean(val_preds == val_df["label"].values))
            print(f"\n>>> Fine-Tuned {self.name} Validation Accuracy: {val_acc * 100:.2f}%")

        return {
            "model_name": self.name,
            "fine_tuned": True,
            "train_samples": len(train_data),
            "epochs": epochs,
            "final_train_acc": float(correct / total),
            "val_accuracy": val_acc
        }

    # --------------------------------------------------------------------------
    # Fallback Linear Probing (Feature Extraction)
    # --------------------------------------------------------------------------
    def _extract_embeddings(self, sentences: List[str], aspects: List[str], batch_size: int = 32) -> np.ndarray:
        transformer = self._get_base_transformer()
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
                outputs = transformer(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_emb = outputs.pooler_output.cpu().numpy()
                else:
                    last_hidden = outputs.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    batch_emb = (torch.sum(last_hidden * mask, 1) / sum_mask).cpu().numpy()
                    
            all_embeddings.append(batch_emb)
        return np.vstack(all_embeddings)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, batch_size: int = 32) -> Dict[str, float]:
        """Linear probing head training (backward compatibility)."""
        print(f"\n--- Training Linear Probing Head for {self.name} on {len(train_df)} samples ---")
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
            print(f"[{self.name}] Head Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        return metrics

    # --------------------------------------------------------------------------
    # Prediction Methods
    # --------------------------------------------------------------------------
    def predict_pair(self, sentence: str, aspect: str) -> Tuple[str, int, Dict[str, float]]:
        if not self.is_trained and self.fine_tuned_model is None:
            raise ValueError(f"Model {self.name} is not trained yet.")

        # 1. Preferred: True End-to-End Fine-Tuned Model
        if self.fine_tuned_model is not None:
            self.fine_tuned_model.eval()
            sent_norm = self.preprocessor.normalize_hinglish(str(sentence))
            asp_norm = self.preprocessor.normalize_hinglish(str(aspect))

            inputs = self.tokenizer(
                [sent_norm],
                [asp_norm],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.fine_tuned_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

            pred_id = int(np.argmax(probs))
            probs_dict = {
                "negative": float(probs[0]),
                "neutral": float(probs[1]),
                "positive": float(probs[2])
            }
            return INV_LABEL_MAP[pred_id], pred_id, probs_dict

        # 2. Fallback: Linear Probing Head
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
        if not self.is_trained and self.fine_tuned_model is None:
            raise ValueError(f"Model {self.name} is not trained yet.")

        # 1. Preferred: True End-to-End Fine-Tuned Model
        if self.fine_tuned_model is not None:
            self.fine_tuned_model.eval()
            sents = [self.preprocessor.normalize_hinglish(s) for s in self._extract_sents(df)]
            asps = [self.preprocessor.normalize_hinglish(a) for a in df["aspect"].astype(str).tolist()]
            
            y_pred = []
            y_probs = []

            for i in range(0, len(sents), batch_size):
                b_sents = sents[i:i + batch_size]
                b_asps = asps[i:i + batch_size]
                inputs = self.tokenizer(
                    b_sents,
                    b_asps,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.fine_tuned_model(**inputs)
                    batch_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                preds = np.argmax(batch_probs, axis=-1)
                y_pred.extend(preds)
                y_probs.extend(batch_probs)

            return np.array(y_pred), np.array(y_probs)

        # 2. Fallback: Linear Probing Head
        sents = self._extract_sents(df)
        asps = df["aspect"].astype(str).tolist()
        emb = self._extract_embeddings(sents, asps, batch_size=batch_size)
        y_pred = self.classifier.predict(emb)
        y_probs = self.classifier.predict_proba(emb)
        return y_pred, y_probs

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------
    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        # If fine-tuned model exists, save transformer weights + tokenizer
        if self.fine_tuned_model is not None:
            self.fine_tuned_model.save_pretrained(dirpath)
            self.tokenizer.save_pretrained(dirpath)
            print(f"Fine-tuned Transformer checkpoint saved to {dirpath}")
        
        # Save classification head info
        head_path = os.path.join(dirpath, "classifier_head.joblib")
        joblib.dump({
            "classifier": self.classifier,
            "model_id": self.model_id,
            "name": self.name,
            "is_trained": self.is_trained,
            "is_fine_tuned": self.is_fine_tuned
        }, head_path)
        print(f"Metadata & classifier checkpoint saved to {dirpath}")

    def load(self, dirpath: str):
        # 1. Check for Fine-Tuned Transformer weights
        has_ft_weights = os.path.exists(os.path.join(dirpath, "config.json")) and (
            os.path.exists(os.path.join(dirpath, "model.safetensors")) or
            os.path.exists(os.path.join(dirpath, "pytorch_model.bin"))
        )
        if has_ft_weights:
            try:
                print(f"Loading Fine-Tuned Transformer from {dirpath}...")
                self.fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(dirpath).to(self.device)
                self.fine_tuned_model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(dirpath)
                self.is_fine_tuned = True
                self.is_trained = True
                print(f"Successfully loaded Fine-Tuned {self.name} into memory!")
            except Exception as e:
                print(f"Notice: Failed to load fine-tuned weights ({e}), falling back to head...")

        # 2. Load linear head if present
        head_path = os.path.join(dirpath, "classifier_head.joblib")
        if os.path.exists(head_path):
            data = joblib.load(head_path)
            self.classifier = data["classifier"]
            self.is_trained = data.get("is_trained", True)
            self.is_fine_tuned = data.get("is_fine_tuned", self.is_fine_tuned)
            print(f"Loaded classifier head from {dirpath}")


class IndicBERTABSAModel(BaseTransformerABSAModel):
    """Model 3: IndicBERT (ai4bharat/IndicBERTv2-MLM-only)"""
    def __init__(self, model_id: str = "ai4bharat/IndicBERTv2-MLM-only", max_length: int = 64, device: str = None):
        super().__init__(model_id=model_id, display_name="IndicBERT", max_length=max_length, device=device)


class MultilingualMiniLMABSAModel(BaseTransformerABSAModel):
    """Model 4: Multilingual MiniLM (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) - Ultra Efficient"""
    def __init__(self, model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", max_length: int = 128, device: str = None):
        super().__init__(model_id=model_id, display_name="Multilingual MiniLM", max_length=max_length, device=device)
