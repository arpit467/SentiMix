import os
import sys
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessor import TextPreprocessor
from src.annotator import MeeshoABSAAnnotator
from src.models.rule_based import RuleBasedABSAModel
from src.models.tfidf_classifier import TFIDFLogisticABSAModel

try:
    from src.models.transformer_classifier import IndicBERTABSAModel, MultilingualMiniLMABSAModel, XLMRoBERTaABSAModel, MuRILABSAModel
except Exception:
    IndicBERTABSAModel = None
    MultilingualMiniLMABSAModel = None
    XLMRoBERTaABSAModel = None
    MuRILABSAModel = None


class ABSAInferenceEngine:
    """
    Unified end-to-end ABSA inference engine:
    1. Extracts aspect terms & categories from arbitrary Hinglish/English text
    2. Runs aspect-sentiment classification using any of the 6 models
    3. Produces structured actionable insights & confidence scores
    """
    def __init__(self, models_dir: str = "results/saved_models"):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.annotator = MeeshoABSAAnnotator()
        self.models = {}
        self._init_models()

    def _init_models(self):
        # 1. Rule-Based
        self.models["Rule-Based Baseline"] = RuleBasedABSAModel()
        
        # 2. TF-IDF + Logistic Regression
        tfidf_path = os.path.join(self.models_dir, "tfidf_logistic.joblib")
        tfidf_model = TFIDFLogisticABSAModel()
        if os.path.exists(tfidf_path):
            try:
                tfidf_model.load(tfidf_path)
            except Exception as e:
                print(f"Warning: Could not load saved TF-IDF model: {e}")
        self.models["TF-IDF + Logistic Regression"] = tfidf_model
        
        # Transformer models placeholders (instantiated on demand or if checkpoints exist)
        self.models["IndicBERT"] = None
        self.models["Multilingual MiniLM"] = None
        self.models["XLM-RoBERTa"] = None
        self.models["MuRIL"] = None

    def get_or_load_transformer(self, model_name: str):
        if self.models.get(model_name) is not None:
            return self.models[model_name]
            
        if model_name == "IndicBERT":
            m = IndicBERTABSAModel()
            ckpt = os.path.join(self.models_dir, "indicbert")
            if os.path.exists(os.path.join(ckpt, "classifier_head.joblib")):
                m.load(ckpt)
            self.models["IndicBERT"] = m
            return m
        elif model_name == "Multilingual MiniLM":
            m = MultilingualMiniLMABSAModel()
            ckpt = os.path.join(self.models_dir, "minilm")
            if os.path.exists(os.path.join(ckpt, "classifier_head.joblib")):
                m.load(ckpt)
            self.models["Multilingual MiniLM"] = m
            return m
        elif model_name == "XLM-RoBERTa":
            m = XLMRoBERTaABSAModel()
            ckpt = os.path.join(self.models_dir, "xlm_roberta")
            if os.path.exists(os.path.join(ckpt, "classifier_head.joblib")):
                m.load(ckpt)
            self.models["XLM-RoBERTa"] = m
            return m
        elif model_name == "MuRIL":
            m = MuRILABSAModel()
            ckpt = os.path.join(self.models_dir, "muril")
            if os.path.exists(os.path.join(ckpt, "classifier_head.joblib")):
                m.load(ckpt)
            self.models["MuRIL"] = m
            return m
        return None

    def _get_aspect_clause_context(self, text: str, aspect: str) -> str:
        """
        Extracts the specific clause context containing the aspect term to prevent sentiment cross-contamination.
        Supports ASCII and Devanagari Unicode word boundaries.
        """
        clauses = self.preprocessor.split_into_clauses(text)
        asp_lower = aspect.lower()
        for clause in clauses:
            pattern = r'(?<![a-zA-Z0-9\u0900-\u097F])' + re.escape(asp_lower) + r'(?![a-zA-Z0-9\u0900-\u097F])'
            if re.search(pattern, clause.lower()):
                return clause
        return text

    def analyze_review(self, review_text: str, model_name: str = "TF-IDF + Logistic Regression", specified_aspects: List[str] = None) -> Dict[str, Any]:
        """
        Extracts aspects and classifies their sentiments in review_text for any custom input.
        """
        if not review_text or not review_text.strip():
            return {
                "original_text": "",
                "cleaned_text": "",
                "is_hinglish": False,
                "model_used": model_name,
                "aspects_found": 0,
                "aspect_breakdown": [],
                "overall_sentiment": "Neutral",
                "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0}
            }

        clean_text = self.preprocessor.normalize_hinglish(review_text)
        is_hing = self.preprocessor.is_hinglish(review_text)
        
        # Step 1: Determine aspect terms
        if specified_aspects and len(specified_aspects) > 0:
            aspect_terms = [a.strip() for a in specified_aspects if a.strip()]
            categories = [self.annotator.keyword_to_category.get(a.lower(), "custom_aspect") for a in aspect_terms]
        else:
            ann = self.annotator.annotate_review(review_text)
            aspect_terms = ann["aspect_terms"]
            categories = ann["aspect_categories"]
            
        # Ensure we always have at least one target if text exists
        if not aspect_terms:
            aspect_terms = ["product"]
            categories = ["product_item"]

        # Step 2: Select Model
        if model_name == "Rule-Based Baseline":
            model = self.models.get("Rule-Based Baseline")
        elif model_name == "TF-IDF + Logistic Regression":
            model = self.models.get("TF-IDF + Logistic Regression")
        else:
            model = self.get_or_load_transformer(model_name)
            
        aspect_results = []
        for idx, asp in enumerate(aspect_terms):
            cat = categories[idx] if idx < len(categories) else self.annotator.keyword_to_category.get(asp.lower(), "general")
            
            # Extract localized clause context for this aspect
            target_context = self._get_aspect_clause_context(review_text, asp)
            
            pol = "neutral"
            probs = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            
            if model is not None and getattr(model, "is_trained", True):
                try:
                    pol, pid, probs = model.predict_pair(target_context, asp)
                except Exception as e:
                    # Fallback to rule-based on clause
                    pol, pid, probs = self.models["Rule-Based Baseline"].predict_pair(target_context, asp)
            else:
                pol, pid, probs = self.models["Rule-Based Baseline"].predict_pair(target_context, asp)
                
            conf = probs.get(pol, max(probs.values()) if probs else 0.8)
            
            aspect_results.append({
                "aspect": asp,
                "category": cat,
                "clause_context": target_context,
                "sentiment": pol,
                "confidence": round(float(conf), 4),
                "probabilities": {k: round(float(v), 4) for k, v in probs.items()}
            })
            
        # Step 3: Overall Sentiment Synthesis
        pos_count = sum(1 for r in aspect_results if r["sentiment"] == "positive")
        neg_count = sum(1 for r in aspect_results if r["sentiment"] == "negative")
        neu_count = sum(1 for r in aspect_results if r["sentiment"] == "neutral")
        
        # Check Sarcasm
        sarcasm_patterns = [
            r"\bnot\b[\s\.!]*$",
            r"\bthanks for the (?:free|gift|pimples)\b",
            r"\bwah\s+(?:kya|kent|bhai|samsung|wipro|puma)\b.*(?:hang|leak|chhale|barbad|bekar|kharab|phati|mazaa|dead)",
            r"\bवाह[,\s]+क्या\b.*(?:खराब|टूटी|उधड़|बर्बाद|छाले|क्वालिटी)",
            r"\bkya baat hai.*service ho toh aisi\b",
            r"\bbrilliant quality control, well done\b"
        ]
        is_sarcastic = any(re.search(p, clean_text.lower()) for p in sarcasm_patterns)
        
        if is_sarcastic:
            overall = "Negative"
            for r in aspect_results:
                r["sentiment"] = "negative"
        elif pos_count > 0 and neg_count > 0:
            overall = "Mixed"
        elif pos_count > 0 and neg_count == 0:
            overall = "Positive"
        elif neg_count > 0 and pos_count == 0:
            overall = "Negative"
        elif neu_count > 0:
            overall = "Neutral"
        else:
            pol, _, _ = self.models["Rule-Based Baseline"].predict_pair(clean_text, "product")
            overall = pol.title()

            
        return {
            "original_text": review_text,
            "cleaned_text": clean_text,
            "is_hinglish": is_hing,
            "model_used": model_name,
            "aspects_found": len(aspect_results),
            "aspect_breakdown": aspect_results,
            "overall_sentiment": overall,
            "sentiment_counts": {
                "positive": pos_count,
                "negative": neg_count,
                "neutral": neu_count
            }
        }


    def analyze_batch(self, df: pd.DataFrame, text_column: str, model_name: str = "TF-IDF + Logistic Regression") -> pd.DataFrame:
        """
        Analyzes a batch DataFrame and returns enriched DataFrame.
        """
        results = []
        for idx, row in df.iterrows():
            text = str(row[text_column])
            res = self.analyze_review(text, model_name=model_name)
            aspect_str = ", ".join([f"{a['aspect']} ({a['sentiment']})" for a in res["aspect_breakdown"]])
            results.append({
                **row.to_dict(),
                "Detected_Hinglish": res["is_hinglish"],
                "Overall_Sentiment": res["overall_sentiment"],
                "Aspects_and_Sentiments": aspect_str,
                "Num_Aspects": res["aspects_found"]
            })
        return pd.DataFrame(results)

if __name__ == "__main__":
    engine = ABSAInferenceEngine()
    test_reviews = [
        "Display bahut vibrant hai 120Hz AMOLED aur battery life 2 days chalti hai lekin camera zoom night me bekar hai",
        "Fabric is super soft and fitting is perfect but colour faded after one wash",
        "Laptop boot time fast hai but keyboard typing noisy hai",
        "Delivery was late and box was damaged but phone is awesome",
        "Ekdum ghatiya product hai mat khareedo",
        "Paisa vasool item hai loved it"
    ]
    for r in test_reviews:
        res = engine.analyze_review(r)
        print(f"\nReview: {r}")
        print(f"Overall: {res['overall_sentiment']} | Aspects: {[(a['aspect'], a['sentiment'], a['confidence']) for a in res['aspect_breakdown']]}")
