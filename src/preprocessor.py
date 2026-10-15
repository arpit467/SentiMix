import re
import emoji
from typing import List, Tuple, Dict, Any

# Common Hinglish spelling variant normalizations
HINGLISH_SPELLING_MAP = {
    # Positive variants
    "acha": "accha", "achha": "accha", "achcha": "accha", "aacha": "accha", "achhi": "acchi",
    "achi": "acchi", "achhii": "acchi", "badiya": "badhiya", "bdhiya": "badhiya", "mast": "mast",
    "shandaar": "shandar", "zabarjast": "jabardast", "zabardast": "jabardast", "lajawav": "lajawab",
    "bohot": "bahut", "bhut": "bahut", "bhot": "bahut", "bht": "bahut", "bahoot": "bahut",
    "pyaraa": "pyara", "piyara": "pyara", "sundar": "sundar", "sundarr": "sundar",
    
    # Negative variants
    "bekaar": "bekar", "bikhar": "bekar", "bkar": "bekar",
    "khrab": "kharab", "khrp": "kharab", "kharap": "kharab", "khrab": "kharab",
    "ghatya": "ghatiya", "gatiya": "ghatiya", "ghatiyaa": "ghatiya",
    "bakwas": "bakwas", "bkwas": "bakwas", "bakwaas": "bakwas",
    "dikkt": "dikkat", "dikat": "dikkat", "prblm": "problem", "prob": "problem",
    "barbaad": "barbad", "barbad": "barbad", "beakar": "bekar",
    
    # Negations & Modifiers
    "nhi": "nahi", "nh": "nahi", "nahin": "nahi", "ni": "nahi", "nae": "nahi",
    "naa": "na", "mat": "mat", "kabhi": "kabhi",
    "jyada": "zyada", "jada": "zyada", "zyda": "zyada",
    "thoda": "thoda", "thora": "thoda", "kam": "kam",
    "bilkul": "bilkul", "blkl": "bilkul",
    
    # Common Nouns & Aspects
    "kapda": "cloth", "kapde": "cloth", "kpda": "cloth", "kpde": "cloth",
    "pese": "paisa", "paise": "paisa", "pesa": "paisa", "rupaye": "paisa", "rs": "rupees",
    "mehanga": "mehenga", "mahnga": "mehenga", "mhnga": "mehenga",
    "sasta": "sasta", "saste": "sasta",
    "khushbu": "fragrance", "khushboo": "fragrance", "khusbu": "fragrance",
    "choti": "chhoti", "chota": "chhota", "bada": "bada", "badi": "badi"
}

# Devanagari to Roman common indicators or keywords
HINDI_ROMAN_INDICATORS = {
    "hai", "hain", "tha", "thi", "the", "nahi", "bahut", "accha", "bekar", "sahi", "mast",
    "badhiya", "kharab", "paisa", "ka", "ki", "ke", "ko", "se", "mein", "par", "bhi",
    "kya", "kyu", "kyun", "ye", "yeh", "wo", "woh", "mera", "meri", "mere", "apna", "apni",
    "hoga", "hogi", "karo", "karna", "mila", "mili", "mile", "lekin", "magar", "aur", "ya",
    "acche", "acchi", "achi", "acha", "achha", "accha", "dikkat", "pasand", "shandar", "kapda",
    "ghar", "khareed", "khareeda", "chalta", "chalti", "chalega", "chalegi", "karana", "padega"
}

class TextPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans review text: removes URLs, HTML tags, duplicate characters,
        translates emojis into text tokens, and normalizes whitespaces while
        preserving Devanagari Unicode characters (U+0900 to U+097F).
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        
        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        
        # Demojize emojis to descriptive tags (e.g., :sparkles:, :thumbs_up:)
        text = emoji.demojize(text, delimiters=(" :", ": "))
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def normalize_hinglish(text: str) -> str:
        """
        Applies Hinglish spelling normalization to canonical tokens while
        preserving Devanagari Hindi words and punctuation.
        """
        cleaned = TextPreprocessor.clean_text(text)
        # Tokenize preserving both Latin words, Devanagari script (\u0900-\u097F), digits, and emoji tags
        tokens = re.findall(r"[a-zA-Z0-9_:]+|[\u0900-\u097F]+|[^\w\s\u0900-\u097F]", cleaned)
        normalized_tokens = []
        for t in tokens:
            t_lower = t.lower()
            if t_lower in HINGLISH_SPELLING_MAP:
                normalized_tokens.append(HINGLISH_SPELLING_MAP[t_lower])
            else:
                normalized_tokens.append(t)
        return " ".join(normalized_tokens)

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Accurately identifies the primary language of the review:
        - "Hindi": Substantial Devanagari Unicode script (>= 3 chars or >= 15% alphabetic content)
        - "Hinglish": Latin script with multiple unambiguous Romanized Hindi tokens (>= 2 distinct words)
        - "English": Standard English vocabulary with no Devanagari and absence of code-mixing
        """
        if not isinstance(text, str) or not text.strip():
            return "English"

        devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
        total_alpha = len(re.findall(r"[a-zA-Z\u0900-\u097F]", text))

        # 1. Hindi script check
        if devanagari_chars >= 3 or (total_alpha > 0 and (devanagari_chars / total_alpha) >= 0.15):
            return "Hindi"

        # 2. English vs Hinglish Roman check
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not tokens:
            return "English"

        unambiguous_hindi = {
            "hai", "hain", "tha", "thi", "nahi", "nahin", "nhi", "bahut", "bohot", "bhut", "bhot",
            "accha", "acchi", "acche", "acha", "achi", "ache", "badiya", "badhiya", "bekar", "bekaar",
            "kharab", "khrab", "ghatiya", "shandar", "mast", "paisa", "paise", "kaise", "apna", "apni", "apne",
            "mera", "meri", "mere", "tera", "teri", "tere", "lekin", "magar", "bhi", "toh", "karo", "karna",
            "karte", "karta", "karti", "kiya", "kiye", "hoga", "hogi", "hoge", "raha", "rahi", "rahe",
            "gaya", "gayi", "gaye", "diya", "diye", "liya", "liye", "chahiye", "dekho", "dekha", "sahi",
            "galat", "bakwas", "khareeda", "pasand", "aaya", "aayi", "aaye", "wala", "wali", "wale", "kuch",
            "sabse", "dikat", "dikkat", "kapda", "kapde", "sasta", "saste", "mehenga", "mehnge", "chhota",
            "chhoti", "bada", "badi", "bilkul", "ekdam", "phir", "kabhi", "kyun", "kyu", "kya", "yeh",
            "woh", "khushbu", "shaadi", "tareef", "ghis", "shor", "mat", "thoda", "thodi", "zyada", "jyada",
            "khareed", "liye"
        }

        matched = [t for t in tokens if t in unambiguous_hindi]
        if len(matched) >= 2 and (len(matched) / len(tokens)) >= 0.06:
            return "Hinglish"
        if len(matched) >= 3:
            return "Hinglish"

        return "English"

    @staticmethod
    def is_hinglish(text: str, threshold: float = 0.08) -> bool:
        """
        Determines if a review is code-mixed Hinglish (or contains Hindi content).
        """
        lang = TextPreprocessor.detect_language(text)
        return lang in ("Hinglish", "Hindi")


    @staticmethod
    def split_into_clauses(text: str) -> List[str]:
        """
        Splits review text into semantic clauses around punctuation and conjunctions in English, Hindi, and Hinglish.
        """
        clean_t = TextPreprocessor.clean_text(text)
        clause_delimiters = r"[.,;!?|\n।]+|\b(?:but|lekin|magar|par|however|though|although|while|aur|and|except|पर|लेकिन|मगर|किंतु|परंतु|और|तथा)\b"
        raw_clauses = re.split(clause_delimiters, clean_t, flags=re.IGNORECASE)
        clauses = [c.strip() for c in raw_clauses if len(c.strip()) > 1]
        return clauses if clauses else [clean_t]

