import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

LABEL_NAMES = ["negative", "neutral", "positive"]

class ModelEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """
        Computes overall and per-class classification metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        
        # Per class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0
        )
        
        return {
            "model_name": model_name,
            "accuracy": acc,
            "precision_macro": p_macro,
            "recall_macro": r_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "per_class_f1": {LABEL_NAMES[i]: f1_per_class[i] for i in range(len(LABEL_NAMES))},
            "confusion_matrix": cm,
            "classification_report": report_dict
        }

    @staticmethod
    def evaluate_model_on_test_set(model: Any, test_df: pd.DataFrame, batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluates a model end-to-end on test_df, including latency and subset breakdowns.
        """
        t0 = time.time()
        if hasattr(model, "predict_batch"):
            if "batch_size" in model.predict_batch.__code__.co_varnames:
                y_pred, y_probs = model.predict_batch(test_df, batch_size=batch_size)
            else:
                y_pred, y_probs = model.predict_batch(test_df)
        else:
            y_pred = []
            for _, r in test_df.iterrows():
                _, pid, _ = model.predict_pair(r["sentence"], r["aspect"])
                y_pred.append(pid)
            y_pred = np.array(y_pred)
            
        latency_total = time.time() - t0
        latency_per_sample_ms = (latency_total / max(len(test_df), 1)) * 1000.0
        
        y_true = test_df["label"].values
        results = ModelEvaluator.evaluate_predictions(y_true, y_pred, model_name=getattr(model, "name", "Model"))
        results["latency_total_sec"] = float(latency_total)
        results["latency_ms_per_sample"] = float(latency_per_sample_ms)
        results["total_test_samples"] = len(test_df)
        
        # 1. Code-mixed (Hinglish) vs Monolingual English subset breakdown
        if "is_hinglish" in test_df.columns:
            hinglish_mask = test_df["is_hinglish"].values.astype(bool)
            if np.sum(hinglish_mask) > 0:
                results["hinglish_subset"] = {
                    "count": int(np.sum(hinglish_mask)),
                    "accuracy": float(accuracy_score(y_true[hinglish_mask], y_pred[hinglish_mask])),
                    "f1_macro": float(f1_score(y_true[hinglish_mask], y_pred[hinglish_mask], average="macro", zero_division=0))
                }
            if np.sum(~hinglish_mask) > 0:
                results["english_subset"] = {
                    "count": int(np.sum(~hinglish_mask)),
                    "accuracy": float(accuracy_score(y_true[~hinglish_mask], y_pred[~hinglish_mask])),
                    "f1_macro": float(f1_score(y_true[~hinglish_mask], y_pred[~hinglish_mask], average="macro", zero_division=0))
                }
                
        # 2. Domain breakdown (Laptop, Restaurant, Meesho, Augmented)
        if "domain" in test_df.columns:
            results["domain_breakdown"] = {}
            for dom in test_df["domain"].unique():
                dom_mask = (test_df["domain"] == dom).values
                results["domain_breakdown"][dom] = {
                    "count": int(np.sum(dom_mask)),
                    "accuracy": float(accuracy_score(y_true[dom_mask], y_pred[dom_mask])),
                    "f1_macro": float(f1_score(y_true[dom_mask], y_pred[dom_mask], average="macro", zero_division=0))
                }
                
        # 3. Top Aspects breakdown
        results["top_aspects_f1"] = {}
        top_aspects = test_df["aspect"].value_counts().head(10).index.tolist()
        for asp in top_aspects:
            asp_mask = (test_df["aspect"] == asp).values
            if np.sum(asp_mask) >= 5:
                results["top_aspects_f1"][asp] = {
                    "count": int(np.sum(asp_mask)),
                    "f1_macro": float(f1_score(y_true[asp_mask], y_pred[asp_mask], average="macro", zero_division=0))
                }
                
        return results

    @staticmethod
    def generate_comparison_table(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a formatted pandas comparison DataFrame from multiple evaluation result dicts.
        """
        rows = []
        for r in all_results:
            rows.append({
                "Model": r["model_name"],
                "Accuracy (%)": round(r["accuracy"] * 100, 2),
                "Macro F1 (%)": round(r["f1_macro"] * 100, 2),
                "Weighted F1 (%)": round(r["f1_weighted"] * 100, 2),
                "Macro Precision (%)": round(r["precision_macro"] * 100, 2),
                "Macro Recall (%)": round(r["recall_macro"] * 100, 2),
                "Hinglish F1 (%)": round(r.get("hinglish_subset", {}).get("f1_macro", 0.0) * 100, 2),
                "Latency (ms/item)": round(r.get("latency_ms_per_sample", 0.0), 2)
            })
        return pd.DataFrame(rows)
