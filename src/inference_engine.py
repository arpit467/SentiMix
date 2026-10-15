import os
import sys
import re
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessor import TextPreprocessor
from src.annotator import MeeshoABSAAnnotator, DynamicAspectExtractor, SARCASM_PATTERNS
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
    1. Extracts aspect terms & categories dynamically from arbitrary Hinglish/English/Hindi text
    2. Runs aspect-sentiment classification using any of the 6 trained models
    3. Produces structured actionable insights, multi-review aggregation, and confidence scores
    """
    def __init__(self, models_dir: str = "results/saved_models"):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.dynamic_extractor = DynamicAspectExtractor()
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
        Extracts aspects dynamically and classifies sentiments in review_text using trained models.
        """
        if not review_text or not str(review_text).strip():
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

        clean_text = self.preprocessor.normalize_hinglish(str(review_text))
        is_hing = self.preprocessor.is_hinglish(str(review_text))
        
        # Step 1: Extract Aspect Terms & Inferred Categories Dynamically
        if specified_aspects and len(specified_aspects) > 0:
            extracted_items = [
                (a.strip(), self.dynamic_extractor.infer_category(a.strip()), self._get_aspect_clause_context(review_text, a.strip()))
                for a in specified_aspects if a.strip()
            ]
        else:
            extracted_items = self.dynamic_extractor.extract_aspects(review_text)

        # Step 2: Select Model
        if model_name == "Rule-Based Baseline":
            model = self.models.get("Rule-Based Baseline")
        elif model_name == "TF-IDF + Logistic Regression":
            model = self.models.get("TF-IDF + Logistic Regression")
        else:
            model = self.get_or_load_transformer(model_name)
            
        aspect_results = []
        for asp, cat, target_context in extracted_items:
            pol = "neutral"
            probs = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
            
            if model is not None and getattr(model, "is_trained", True):
                try:
                    pol, pid, probs = model.predict_pair(target_context, asp)
                except Exception:
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
        is_sarcastic = any(re.search(p, clean_text.lower()) for p in SARCASM_PATTERNS)
        
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

    def analyze_multiple_reviews(self, reviews: List[str], model_name: str = "TF-IDF + Logistic Regression") -> Dict[str, Any]:
        """
        Analyzes a list of arbitrary customer reviews (1 or multiple).
        Aggregates fine-grained ABSA metrics, creates visual distribution data,
        and saves an exportable CSV report.
        """
        t0 = time.time()
        valid_reviews = [r.strip() for r in reviews if r and str(r).strip()]
        if not valid_reviews:
            return {
                "total_reviews": 0,
                "total_aspects": 0,
                "avg_aspects_per_review": 0,
                "hinglish_percentage": 0,
                "pure_lang_percentage": 0,
                "avg_confidence": 0,
                "net_sentiment_score": 0,
                "sentiment_counts": {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0},
                "sentiment_percentages": {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0},
                "top_categories": [],
                "top_aspect_terms": [],
                "aspect_records": [],
                "reviews_breakdown": [],
                "download_url": "",
                "latency_sec": 0
            }

        total_reviews = len(valid_reviews)
        aspect_records = []
        category_counts = {}
        aspect_counts = {}
        overall_sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
        hinglish_count = 0
        total_confidence = 0.0
        aspects_count_total = 0
        reviews_breakdown = []
        enriched_rows = []

        for idx, text in enumerate(valid_reviews):
            res = self.analyze_review(text, model_name=model_name)
            reviews_breakdown.append(res)
            
            if res["is_hinglish"]:
                hinglish_count += 1
                
            overall = res["overall_sentiment"]
            overall_sentiments[overall] = overall_sentiments.get(overall, 0) + 1
            aspects_in_review = res["aspect_breakdown"]
            aspects_count_total += len(aspects_in_review)
            
            aspect_summary_str = "; ".join([f"{a['aspect']} ({a['sentiment'].upper()})" for a in aspects_in_review])
            
            enriched_rows.append({
                "Review_Index": idx + 1,
                "Customer_Review": text,
                "Detected_Language": "Hinglish" if res["is_hinglish"] else "English",
                "Overall_Sentiment": overall,
                "Aspects_Extracted": aspect_summary_str if aspect_summary_str else "Product-level",
                "Num_Aspects": len(aspects_in_review)
            })
            
            for asp_info in aspects_in_review:
                asp = asp_info["aspect"].lower().strip()
                cat = asp_info.get("category", "product_feature")
                sent = asp_info["sentiment"].lower()
                conf = asp_info.get("confidence", 0.8)
                total_confidence += conf
                
                # Aspect term counts
                if asp not in aspect_counts:
                    aspect_counts[asp] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
                aspect_counts[asp][sent] = aspect_counts[asp].get(sent, 0) + 1
                aspect_counts[asp]["total"] += 1
                
                # Category counts
                cat_clean = cat.replace("_", " ").title()
                if cat_clean not in category_counts:
                    category_counts[cat_clean] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
                category_counts[cat_clean][sent] = category_counts[cat_clean].get(sent, 0) + 1
                category_counts[cat_clean]["total"] += 1
                
                aspect_records.append({
                    "row_index": idx + 1,
                    "review": text[:120] + ("..." if len(text) > 120 else ""),
                    "aspect": asp,
                    "category": cat_clean,
                    "sentiment": sent,
                    "confidence": round(float(conf), 4),
                    "is_hinglish": res["is_hinglish"],
                    "clause": asp_info.get("clause_context", "")
                })

        # Top Categories & Aspects
        top_categories = sorted(category_counts.items(), key=lambda x: x[1]["total"], reverse=True)[:8]
        formatted_top_categories = [
            {"category": k, "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0), "total": v["total"]}
            for k, v in top_categories
        ]

        top_aspect_terms = sorted(aspect_counts.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
        formatted_top_aspects = [
            {"aspect": k, "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0), "total": v["total"]}
            for k, v in top_aspect_terms
        ]

        avg_conf = round(total_confidence / aspects_count_total * 100, 1) if aspects_count_total > 0 else 85.0
        hinglish_pct = round(hinglish_count / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        pos_pct = round(overall_sentiments.get("Positive", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        neg_pct = round(overall_sentiments.get("Negative", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        neu_pct = round(overall_sentiments.get("Neutral", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        mixed_pct = round(overall_sentiments.get("Mixed", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        net_sentiment = round(pos_pct - neg_pct, 1)

        # Save report CSV
        os.makedirs("results", exist_ok=True)
        out_filename = f"absa_analyzed_{int(time.time())}.csv"
        out_path = os.path.join("results", out_filename)
        pd.DataFrame(enriched_rows).to_csv(out_path, index=False, encoding="utf-8")

        latency_sec = round(time.time() - t0, 2)

        return {
            "total_reviews": total_reviews,
            "total_aspects": aspects_count_total,
            "avg_aspects_per_review": round(aspects_count_total / total_reviews, 2) if total_reviews > 0 else 0,
            "hinglish_percentage": hinglish_pct,
            "pure_lang_percentage": round(100.0 - hinglish_pct, 1),
            "avg_confidence": avg_conf,
            "net_sentiment_score": net_sentiment,
            "sentiment_counts": overall_sentiments,
            "sentiment_percentages": {"Positive": pos_pct, "Negative": neg_pct, "Neutral": neu_pct, "Mixed": mixed_pct},
            "top_categories": formatted_top_categories,
            "top_aspect_terms": formatted_top_aspects,
            "download_url": f"/api/download/{out_filename}",
            "aspect_records": aspect_records[:250],
            "reviews_breakdown": reviews_breakdown[:100],
            "latency_sec": latency_sec
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
