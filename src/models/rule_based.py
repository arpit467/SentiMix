import re
import emoji
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.preprocessor import TextPreprocessor

POSITIVE_LEXICON = {
    "good", "nice", "great", "best", "awesome", "super", "superb", "excellent",
    "amazing", "love", "loved", "loving", "perfect", "perfectly", "beautiful", "pyara",
    "pyari", "mast", "badhiya", "accha", "acha", "achha", "achi", "achhi", "acche",
    "sahi", "fabulous", "wonderful", "satisfied", "satisfactory", "happy", "comfortable",
    "comfy", "soft", "affordable", "worth", "worthit", "paisa vasool", "paisa wasool",
    "liked", "like", "recommend", "must buy", "outstanding", "top notch", "flawless",
    "original", "genuine", "dashing", "cute", "pretty", "jabardast", "shandar", "gazab",
    "kamaal", "lajawab", "mindblowing", "favourite", "favorite", "smooth", "shine", "glow",
    "fast", "clear", "crisp", "heavy", "punchy", "sharp", "long", "reliable", "easy", "vibrant"
}

NEGATIVE_LEXICON = {
    "bad", "worst", "poor", "terrible", "horrible", "pathetic", "useless", "waste",
    "bakwas", "bekar", "bekaar", "kharab", "khrab", "bura", "buri", "ghatiya", "fake",
    "duplicate", "damaged", "damage", "cheap", "rough", "disappointed", "disappointing",
    "faulty", "defective", "loose", "tight", "chhota", "chhoti", "bada", "badi", "late",
    "delay", "missing", "faded", "dull", "rash", "allergy", "irritation", "smelly",
    "third class", "waste of money", "paisa barbad", "fraud", "scam", "uncomfortable",
    "not good", "never buy", "hard", "transparent", "faded", "tear", "torn", "slow",
    "hang", "garam", "heating", "overheat", "overheating", "loud", "noise", "wash", "mat khareedo", "mat lena"
}

NEGATION_TERMS = {
    "not", "no", "nahi", "nahin", "na", "never", "without", "bina", "hardly", "barely", "kam", "don't", "dont"
}

DOUBLE_NEG_POSITIVE = {
    "not bad", "not bad at all", "bura nahi", "bura nhi", "kharab nahi", "kharab nhi", "bekar nahi", "bekar nhi", "not poor"
}

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class RuleBasedABSAModel:
    """
    Model 1: Rule-Based Lexicon and Syntactic Proximity Baseline.
    Evaluates polarity of a target aspect term using window proximity,
    negation handling, and code-mixed sentiment dictionaries.
    """
    def __init__(self, window_size: int = 6):
        self.name = "Rule-Based Baseline"
        self.window_size = window_size
        self.preprocessor = TextPreprocessor()

    def predict_pair(self, sentence: str, aspect: str) -> Tuple[str, int, Dict[str, float]]:
        """
        Predicts sentiment polarity for a (sentence, aspect) pair.
        Returns: (polarity_name, label_id, confidence_probabilities)
        """
        clean_s = self.preprocessor.normalize_hinglish(sentence).lower()
        aspect_clean = self.preprocessor.normalize_hinglish(aspect).lower()
        
        words = re.findall(r"[a-z0-9_:]+|[^\w\s]", clean_s)
        asp_words = re.findall(r"[a-z0-9_:]+", aspect_clean)
        
        # Find index of aspect occurrence
        asp_idx = -1
        if asp_words:
            first_asp_word = asp_words[0]
            for idx, w in enumerate(words):
                if w == first_asp_word:
                    asp_idx = idx
                    break
                    
        # Define local window around aspect (or whole sentence if aspect is not found)
        if asp_idx != -1:
            start = max(0, asp_idx - self.window_size)
            end = min(len(words), asp_idx + len(asp_words) + self.window_size)
            window_words = words[start:end]
            window_text = " ".join(window_words)
        else:
            window_words = words
            window_text = clean_s

        # Explicit double negation check ("not bad" -> positive)
        for dn in DOUBLE_NEG_POSITIVE:
            if dn in window_text:
                return "positive", LABEL_MAP["positive"], {"negative": 0.10, "neutral": 0.20, "positive": 0.70}

        # Negation check
        has_neg = any(neg in window_words for neg in NEGATION_TERMS) or ("nahi" in window_text) or ("not" in window_text) or ("mat " in window_text)
        
        pos_score = sum(1.0 for w in window_words if w in POSITIVE_LEXICON)
        neg_score = sum(1.0 for w in window_words if w in NEGATIVE_LEXICON)
        
        # Emoji boost
        if ":thumbs_up:" in window_text or ":heart:" in window_text or ":fire:" in window_text or ":star:" in window_text:
            pos_score += 1.5
        if ":thumbs_down:" in window_text or ":disappointed:" in window_text or ":rage:" in window_text or ":crying:" in window_text:
            neg_score += 1.5

        if has_neg:
            if pos_score > 0 and neg_score == 0:
                # "not good", "accha nahi hai" -> negative
                pred = "negative"
                probs = {"negative": 0.85, "neutral": 0.10, "positive": 0.05}
            elif neg_score > 0:
                # "ghatiya product mat khareedo" -> negative
                pred = "negative"
                probs = {"negative": 0.90, "neutral": 0.07, "positive": 0.03}
            else:
                pred = "negative"
                probs = {"negative": 0.70, "neutral": 0.20, "positive": 0.10}
        else:
            if pos_score > neg_score:
                pred = "positive"
                probs = {"negative": 0.05, "neutral": 0.15, "positive": 0.80}
            elif neg_score > pos_score:
                pred = "negative"
                probs = {"negative": 0.80, "neutral": 0.15, "positive": 0.05}
            else:
                pred = "neutral"
                probs = {"negative": 0.20, "neutral": 0.60, "positive": 0.20}

        return pred, LABEL_MAP[pred], probs

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs prediction over DataFrame with 'sentence'/'clean_sentence' and 'aspect' columns.
        Returns: (y_pred, y_probs)
        """
        y_pred = []
        y_probs = []
        for _, row in df.iterrows():
            sent = row.get("clean_sentence", row["sentence"])
            asp = row["aspect"]
            pred_label_name, pred_id, probs = self.predict_pair(sent, asp)
            y_pred.append(pred_id)
            y_probs.append([probs["negative"], probs["neutral"], probs["positive"]])
        return np.array(y_pred), np.array(y_probs)
