import os
import sys
import re
import ast
import pandas as pd
from typing import List, Tuple, Dict, Any, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessor import TextPreprocessor, HINGLISH_SPELLING_MAP

# Comprehensive Multi-Domain Aspect Term Dictionary across E-Commerce, Tech, Fashion, Food, etc.
ASPECT_CATEGORIES = {
    # Electronics & Hardware
    "battery_power": [
        "battery", "backup", "charging", "charger", "battery life", "battery backup", "charge",
        "standby", "mah", "drain", "draining", "power", "adapter", "fast charging", "overheating"
    ],
    "camera_photography": [
        "camera", "photo", "picture", "pictures", "lens", "selfie", "portrait", "night mode",
        "video", "clarity", "zoom", "sensors", "megapixels", "mp", "shots", "images", "image quality"
    ],
    "display_screen": [
        "display", "screen", "amoled", "oled", "refresh rate", "brightness", "resolution",
        "colors", "viewing angles", "panel", "touch", "touchscreen", "120hz", "90hz", "hd"
    ],
    "performance_speed": [
        "performance", "speed", "processor", "ram", "storage", "gaming", "lag", "hanging",
        "smooth", "fps", "heating", "multitasking", "boot time", "ssd", "gpu", "cpu", "thermal"
    ],
    "audio_sound": [
        "sound", "audio", "bass", "volume", "speakers", "mic", "microphone", "clarity",
        "noise cancellation", "anc", "earbuds", "headphones", "treble", "voice", "voice recognition"
    ],
    "connectivity_hardware": [
        "bluetooth", "wifi", "network", "range", "connectivity", "ports", "usb", "keyboard",
        "keypad", "touchpad", "trackpad", "hinge", "buttons", "fingerprint", "sensor"
    ],
    # Wearables & Comfort
    "comfort_ergonomics": [
        "comfort", "comfortable", "uncomfortable", "strap", "belt", "weight", "lightweight",
        "heavy", "grip", "wear", "fitting", "ergonomic", "softness", "scratch", "durability"
    ],
    # Fashion, Apparel & Accessories
    "fabric_material": [
        "fabric", "material", "cloth", "stuff", "silk", "cotton", "khadi", "rayon",
        "georgette", "velvet", "polyester", "wool", "net", "chiffon", "texture", "kapda", "kapde"
    ],
    "fit_size": [
        "fitting", "fit", "size", "length", "waist", "loose", "tight", "pant fitting",
        "kurta fitting", "sleeves", "height", "perfect fit", "dimension", "shape", "measurement"
    ],
    "design_appearance": [
        "design", "look", "looks", "color", "colour", "shade", "print", "embroidery",
        "work", "pattern", "style", "appearance", "shine", "glow", "pigment", "pigmentation", "finish", "dial"
    ],
    "quality_build": [
        "quality", "durability", "build", "stitching", "interlock", "pocket", "pockets",
        "zip", "buttons", "durable", "finishing", "stitching quality", "body", "frame"
    ],
    # Commercial & Services
    "price_value": [
        "price", "cost", "paisa", "paise", "rate", "worth", "affordable", "expensive",
        "sasta", "mehenga", "budget", "value for money", "cheap", "overpriced", "deal", "discount"
    ],
    "delivery_service": [
        "delivery", "shipping", "courier", "package", "packaging", "box", "deliver",
        "packing", "time pe", "late", "delayed", "damage", "sealed", "safety", "seller", "meesho", "flipkart", "amazon", "service", "customer support", "support"
    ],
    # Beauty & Personal Care
    "scent_fragrance": [
        "fragrance", "smell", "scent", "khushbu", "khushboo", "odour", "perfume", "aroma"
    ],
    "skin_beauty_effect": [
        "skin", "face", "hair", "moisture", "moisturizer", "glow", "softness", "rash",
        "allergy", "irritation", "hydration", "pimple", "dark spots", "results", "effect"
    ],
    # Food & Hospitality
    "food_taste": [
        "food", "taste", "flavor", "flavour", "dish", "dishes", "curry", "paneer", "chicken",
        "biryani", "pizza", "burger", "dessert", "drink", "portion", "freshness"
    ],
    "ambience_service": [
        "ambience", "service", "staff", "waiter", "hospitality", "seating", "table",
        "music", "atmosphere", "cleanliness", "hygiene", "vibe"
    ],
    # Product Categories
    "product_item": [
        "phone", "mobile", "laptop", "earphones", "watch", "smartwatch", "tv", "camera",
        "suit", "dress", "kurti", "kurta", "dupatta", "pant", "pants", "shawl", "chunri",
        "saree", "top", "palazzo", "cream", "lipstick", "serum", "kajal", "product", "item", "set", "device"
    ]
}

# Positive sentiment lexicons (English + Hinglish Transliterated)
POSITIVE_LEXICON = {
    "good", "nice", "great", "best", "awesome", "super", "superb", "excellent",
    "amazing", "love", "loved", "loving", "perfect", "perfectly", "beautiful", "pyara",
    "pyari", "mast", "badhiya", "accha", "acha", "achha", "achi", "achhi", "acche",
    "sahi", "fabulous", "wonderful", "satisfied", "satisfactory", "happy", "comfortable",
    "comfy", "soft", "affordable", "worth", "worthit", "paisa vasool", "paisa wasool",
    "liked", "like", "recommend", "must buy", "outstanding", "top notch", "flawless",
    "original", "genuine", "dashing", "cute", "pretty", "jabardast", "shandar", "gazab",
    "kamaal", "lajawab", "mindblowing", "favourite", "favorite", "smooth", "shine", "glow",
    "fast", "sharp", "crisp", "clear", "premium", "solid", "vibrant", "loud", "punchy", "reliable"
}

# Negative sentiment lexicons (English + Hinglish Transliterated)
NEGATIVE_LEXICON = {
    "bad", "worst", "poor", "terrible", "horrible", "pathetic", "useless", "waste",
    "bakwas", "bekar", "bekaar", "kharab", "khrab", "bura", "buri", "ghatiya", "fake",
    "duplicate", "damaged", "damage", "cheap", "rough", "disappointed", "disappointing",
    "faulty", "defective", "loose", "tight", "chhota", "chhoti", "bada", "badi", "late",
    "delay", "missing", "faded", "dull", "rash", "allergy", "irritation", "smelly",
    "third class", "waste of money", "paisa barbad", "fraud", "scam", "uncomfortable",
    "not good", "never buy", "hard", "transparent", "tear", "torn", "slow", "blurry",
    "noisy", "hang", "laggy", "overpriced", "expensive", "drain", "draining", "heating"
}

NEGATION_TOKENS = {
    "not", "no", "nahi", "nahin", "na", "mat", "never", "without", "bina", "hardly", "barely", "kam", "don't", "dont"
}

# Strict Stop Words & Intensifiers (MUST NEVER be treated as aspect terms)
STOP_WORDS_AND_MODIFIERS = {
    "is", "was", "are", "the", "a", "an", "and", "or", "to", "in", "on", "for", "with",
    "hai", "hain", "tha", "thi", "the", "ho", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "bhi", "bahut", "bohot", "bhut", "bhot", "very", "kaafi", "kafi", "kfi", "kafi", "thoda",
    "thodi", "thora", "zyada", "jada", "jyada", "bilkul", "ekdum", "sirf", "toh", "to", "so",
    "too", "really", "quite", "ab", "jab", "tab", "kuch", "khas", "laga", "lagi", "lage",
    "lagta", "lagti", "lagte", "raha", "rahi", "rahe", "hona", "hone", "hota", "hoti", "hote",
    "karo", "karna", "kare", "wali", "wala", "wale", "yeh", "ye", "woh", "wo", "mera", "meri",
    "mere", "apna", "apni", "apne", "after", "before", "while", "then", "just", "also", "even",
    "but", "par", "lekin", "magar", "hai", "tha", "thi", "bhai", "yaar", "plz", "please"
}

class MeeshoABSAAnnotator:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # Build flattened keyword lookup sorted by length descending
        self.keyword_to_category = {}
        for cat, kws in ASPECT_CATEGORIES.items():
            for kw in kws:
                self.keyword_to_category[kw.lower()] = cat
        self.sorted_keywords = sorted(self.keyword_to_category.keys(), key=lambda x: len(x), reverse=True)

    def _determine_clause_sentiment(self, clause: str, overall_rating: float = 4.0) -> str:
        clause_clean = clause.lower()
        words = re.findall(r"[a-z0-9_:]+", clause_clean)
        
        emoji_pos = any(e in clause_clean for e in [":thumbs_up:", ":heart:", ":sparkles:", ":fire:", ":smiling_", ":star:", ":folded_hands:"])
        emoji_neg = any(e in clause_clean for e in [":thumbs_down:", ":crying_", ":rage:", ":disappointed:", ":broken_heart:", ":weary:"])
        
        has_negation = any(neg in words for neg in NEGATION_TOKENS) or ("nahi" in clause_clean) or ("not" in clause_clean) or ("mat " in clause_clean)
        
        pos_count = sum(1 for w in words if w in POSITIVE_LEXICON) + (1 if emoji_pos else 0)
        neg_count = sum(1 for w in words if w in NEGATIVE_LEXICON) + (1 if emoji_neg else 0)
        
        if has_negation:
            if pos_count > 0 and neg_count == 0:
                return "negative"
            elif neg_count > 0:
                return "negative"
            else:
                return "negative"
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            if overall_rating >= 4.0:
                return "positive"
            elif overall_rating <= 2.0:
                return "negative"
            else:
                return "neutral"

    def _extract_candidate_aspects(self, clause: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", clause.lower())
        candidates = []
        for w in words:
            if w in self.keyword_to_category:
                candidates.append(w)
            elif (w not in POSITIVE_LEXICON and 
                  w not in NEGATIVE_LEXICON and 
                  w not in NEGATION_TOKENS and 
                  w not in STOP_WORDS_AND_MODIFIERS):
                if len(w) >= 3 and not any(ch.isdigit() for ch in w):
                    candidates.append(w)
        return candidates

    def annotate_review(self, text: str, rating: float = 4.0, mapped_sentiment: str = "") -> Dict[str, Any]:
        clean_norm = self.preprocessor.normalize_hinglish(text).lower()
        clauses = self.preprocessor.split_into_clauses(clean_norm)
        
        aspect_terms = []
        at_polarity = []
        aspect_categories = []
        found_terms = set()
        
        # 1. First Pass: Check known multi-domain aspect keywords across clauses
        for clause in clauses:
            clause_sent = self._determine_clause_sentiment(clause, rating)
            clause_matched = False
            for kw in self.sorted_keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, clause):
                    if kw not in found_terms and kw not in STOP_WORDS_AND_MODIFIERS:
                        found_terms.add(kw)
                        aspect_terms.append(kw)
                        at_polarity.append(clause_sent)
                        aspect_categories.append(self.keyword_to_category[kw])
                        clause_matched = True
                        
            # If a clause has strong sentiment but no known keyword, extract non-stopword candidate noun
            if not clause_matched and len(clause.strip()) > 3:
                candidates = self._extract_candidate_aspects(clause)
                for cand in candidates[:1]:
                    if cand not in found_terms and cand not in STOP_WORDS_AND_MODIFIERS:
                        found_terms.add(cand)
                        aspect_terms.append(cand)
                        at_polarity.append(clause_sent)
                        aspect_categories.append(self.keyword_to_category.get(cand, "custom_aspect"))
        
        # 2. Fallback if still no aspect term extracted
        if not aspect_terms and len(clean_norm) > 2:
            default_aspect = "product"
            fallback_pol = self._determine_clause_sentiment(clean_norm, rating)
            if mapped_sentiment and isinstance(mapped_sentiment, str):
                ms = mapped_sentiment.lower()
                if "good" in ms:
                    fallback_pol = "positive"
                elif "bad" in ms:
                    fallback_pol = "negative"
                elif "avg" in ms or "average" in ms:
                    fallback_pol = "neutral"
            aspect_terms.append(default_aspect)
            at_polarity.append(fallback_pol)
            aspect_categories.append("product_item")
            
        return {
            "sentence": text,
            "cleaned_sentence": clean_norm,
            "aspect_terms": aspect_terms,
            "at_polarity": at_polarity,
            "aspect_categories": aspect_categories,
            "num_aspect_terms": len(aspect_terms),
            "is_hinglish": self.preprocessor.is_hinglish(text)
        }

    def annotate_meesho_dataset(self, input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
        print(f"Loading Meesho reviews from {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        df = df.dropna(subset=["comments"]).reset_index(drop=True)
        
        annotated_records = []
        for idx, row in df.iterrows():
            text = str(row["comments"])
            rating = float(row["ratings"]) if pd.notna(row.get("ratings")) else 4.0
            mapped_sent = str(row["mapped_sentiment"]) if pd.notna(row.get("mapped_sentiment")) else ""
            
            ann = self.annotate_review(text, rating, mapped_sent)
            annotated_records.append({
                "id": row.get("id", idx + 1),
                "sentence": ann["sentence"],
                "cleaned_sentence": ann["cleaned_sentence"],
                "aspect_terms": str(ann["aspect_terms"]),
                "at_polarity": str(ann["at_polarity"]),
                "aspect_categories": str(ann["aspect_categories"]),
                "#aspect terms": ann["num_aspect_terms"],
                "is_hinglish": ann["is_hinglish"],
                "product_category": row.get("category", "General"),
                "rating": rating
            })
            
        annotated_df = pd.DataFrame(annotated_records)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        annotated_df.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Successfully annotated {len(annotated_df)} Meesho reviews -> Saved to {output_csv_path}")
        return annotated_df
