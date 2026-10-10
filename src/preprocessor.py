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
    def is_hinglish(text: str, threshold: float = 0.08) -> bool:
        """
        Determines if a review contains Hindi or Hindi-English code-mixed content.
        """
        if not isinstance(text, str):
            return False
        
        # Check Devanagari characters
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
        if has_devanagari:
            return True
        
        # Check Romanized Hindi vocabulary
        tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
        if not tokens:
            return False
        
        matched_hindi = tokens.intersection(HINDI_ROMAN_INDICATORS)
        ratio = len(matched_hindi) / max(len(tokens), 1)
        return len(matched_hindi) >= 1 or ratio >= threshold

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

