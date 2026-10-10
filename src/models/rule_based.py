import re
import emoji
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.preprocessor import TextPreprocessor

POSITIVE_LEXICON = {
    # English
    "good", "nice", "great", "best", "awesome", "super", "superb", "excellent",
    "amazing", "love", "loved", "loving", "perfect", "perfectly", "beautiful",
    "fabulous", "wonderful", "satisfied", "satisfactory", "happy", "comfortable",
    "comfy", "soft", "affordable", "worth", "worthit", "liked", "like", "recommend",
    "recommended", "must buy", "outstanding", "top notch", "flawless", "original",
    "genuine", "dashing", "cute", "pretty", "smooth", "shine", "glow", "fast",
    "sharp", "crisp", "clear", "premium", "solid", "vibrant", "loud", "punchy",
    "reliable", "refreshing", "gentle", "sturdy", "durable", "snug", "breathable",
    "non-sticky", "stretchy", "opaque", "crispy", "easy", "tidy", "neat", "quiet",
    "silent", "prompt", "polite", "value for money", "paisa vasool", "paisa wasool",
    "recomended", "smudge proof", "bright", "super warm", "warm",
    
    # Hinglish
    "pyara", "pyari", "mast", "badhiya", "accha", "acha", "achha", "achi", "achhi",
    "acche", "sahi", "jabardast", "shandar", "shandaar", "gazab", "kamaal", "lajawab",
    "mindblowing", "favourite", "favorite", "tikau", "dhamakedaar", "badiya", "khush",
    "pasand", "mazedaar",
    
    # Devanagari Hindi
    "अच्छा", "अच्छी", "अच्छे", "बढ़िया", "शानदार", "सुंदर", "मस्त", "जबरदस्त", "बेहतरीन",
    "प्यारा", "प्यारी", "सस्ता", "सस्ते", "संतुष्ट", "नरम", "आरामदायक", "साफ", "सही",
    "किफायती", "पैसे वसूल", "दमदार", "तेज़", "वाजिब", "पसंद", "चमक", "सुविधाजनक",
    "मुलायम", "मजबूत", "सुरक्षित", "विनम्र", "स्मूद", "बेस्ट", "सुपर", "हल्का", "हल्के",
    "चटक", "कूल", "शीतल", "गर्म"
}

NEGATIVE_LEXICON = {
    # English
    "bad", "worst", "poor", "terrible", "horrible", "pathetic", "useless", "waste",
    "cheap", "rough", "disappointed", "disappointing", "faulty", "defective", "loose",
    "tight", "late", "delay", "delayed", "missing", "faded", "fades", "fade", "fading",
    "dull", "duller", "rash", "rashes", "allergy", "irritation", "smelly", "third class",
    "waste of money", "fraud", "scam", "uncomfortable", "not good", "never buy", "hard",
    "transparent", "tear", "torn", "slow", "blurry", "noisy", "hang", "hangs", "laggy",
    "overpriced", "expensive", "drain", "drains", "draining", "heating", "leaky", "leak",
    "leaks", "leaking", "stopped working", "detached", "breakouts", "pimples", "dead on arrival",
    "dead", "fake", "headache", "awful", "unreliable", "itchy", "unbearable", "slippery",
    "crushed", "flimsy", "patchy", "shedding", "hurts", "bite", "garbage", "dent",
    "misprinted", "burns", "annoying", "cakey", "chipped", "chipping", "smudge", "disconnecting",
    
    # Hinglish
    "bakwas", "bkwas", "bekar", "bekaar", "kharab", "khrab", "bura", "buri", "ghatiya",
    "paisa barbad", "chhota", "chhoti", "bada", "badi", "garam", "shor", "chhale",
    "chhole", "dukhte", "sasti", "jhanjhat", "dhili", "patla", "toot", "phati", "thandi",
    "dikkat", "mat lena", "mat khareedo", "barbad", "barbaad",
    
    # Devanagari Hindi
    "खराब", "घटिया", "बेकार", "बर्बाद", "बकवास", "बुरा", "बुरी", "टूटा", "टूटी", "फटा",
    "फटी", "लीक", "लीकेज", "धीमी", "धीमा", "महंगा", "महंगी", "दर्द", "मुसीबत", "परेशान",
    "झंझट", "रैश", "चिपचिपा", "चिपचिपापन", "फिसलन", "कमज़ोर", "अविश्वसनीय", "मुहांसे",
    "उधड़", "झड़ना", "सिरदर्द", "चोट", "छाले", "शोर", "फीका", "नकली", "धोखा", "भारी",
    "बदबू", "देर", "ढीला", "ढीली", "कमजोर"
}

NEUTRAL_LEXICON = {
    "theek theek", "theek", "average", "normal", "nothing special", "okay", "okay okay",
    "okay type", "chalta hai", "theek thaak", "gets the job done", "औसत", "सामान्य", "ठीक-ठाक",
    "ठीक ठाक", "न बुरा न अच्छा", "neither great nor bad", "neither great or bad", "does the job", "expected", "theek hai"
}


NEGATION_TERMS = {
    "not", "no", "nahi", "nahin", "na", "never", "without", "bina", "hardly", "barely", "kam",
    "don't", "dont", "नहीं", "ना", "न", "मत", "बिना"
}

DOUBLE_NEG_POSITIVE = {
    "not bad", "not bad at all", "bura nahi", "bura nhi", "kharab nahi", "kharab nhi",
    "bekar nahi", "bekar nhi", "not poor", "no issues", "no problem", "zero damage",
    "no complaints", "not drying at all", "no dryness", "no foot pain", "no white cast", "not see-through"
}

SARCASM_PATTERNS = [
    r"\bnot\b[\s\.!]*$",
    r"\bthanks for the (?:free|gift|pimples)\b",
    r"\bwah\s+(?:kya|kent|bhai|samsung|wipro|puma)\b.*(?:hang|leak|chhale|barbad|bekar|kharab|phati|mazaa|dead)",
    r"\bवाह[,\s]+क्या\b.*(?:खराब|टूटी|उधड़|बर्बाद|छाले|क्वालिटी)",
    r"\bkya baat hai.*service ho toh aisi\b",
    r"\bbrilliant quality control, well done\b"
]


LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

class RuleBasedABSAModel:
    """
    Model 1: Rule-Based Lexicon and Syntactic Proximity Baseline.
    Evaluates polarity of a target aspect term using window proximity,
    negation handling, code-mixed sentiment dictionaries, and Devanagari Hindi support.
    """
    def __init__(self, window_size: int = 8):
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
        
        # Regex tokenization supporting ASCII and Devanagari Unicode
        words = re.findall(r"[a-z0-9_:]+|[\u0900-\u097F]+|[^\w\s\u0900-\u097F]", clean_s)
        asp_words = re.findall(r"[a-z0-9_:]+|[\u0900-\u097F]+", aspect_clean)
        
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

        # Explicit Sarcasm Check
        for spat in SARCASM_PATTERNS:
            if re.search(spat, clean_s, re.IGNORECASE):
                return "negative", LABEL_MAP["negative"], {"negative": 0.88, "neutral": 0.08, "positive": 0.04}

        # Explicit double negation check ("not bad" -> positive)
        for dn in DOUBLE_NEG_POSITIVE:
            if dn in window_text:
                return "positive", LABEL_MAP["positive"], {"negative": 0.05, "neutral": 0.15, "positive": 0.80}

        # Neutral expressions check
        for neu in NEUTRAL_LEXICON:
            if neu in window_text:
                return "neutral", LABEL_MAP["neutral"], {"negative": 0.15, "neutral": 0.70, "positive": 0.15}

        # Negation check
        has_neg = any(neg in window_words for neg in NEGATION_TERMS) or ("nahi" in window_text) or ("not" in window_text) or ("mat " in window_text) or ("नहीं" in window_text)
        
        pos_score = sum(1.0 for w in window_words if w in POSITIVE_LEXICON)
        neg_score = sum(1.0 for w in window_words if w in NEGATIVE_LEXICON)
        
        # Emoji boost
        if ":thumbs_up:" in window_text or ":heart:" in window_text or ":fire:" in window_text or ":star:" in window_text:
            pos_score += 1.5
        if ":thumbs_down:" in window_text or ":disappointed:" in window_text or ":rage:" in window_text or ":crying:" in window_text:
            neg_score += 1.5

        if has_neg:
            if pos_score > 0 and neg_score == 0:
                pred = "negative"
                probs = {"negative": 0.85, "neutral": 0.10, "positive": 0.05}
            elif neg_score > 0:
                if any(good_neg in window_text for good_neg in ["no pain", "no foot pain", "no white cast", "no complaints", "no issues", "zero damage", "no problem", "awaaz bilkul nahi", "awaaz nahi", "noise bilkul nahi", "no dryness"]):
                    pred = "positive"
                    probs = {"negative": 0.05, "neutral": 0.15, "positive": 0.80}
                else:
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
