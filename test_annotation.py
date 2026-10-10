import pandas as pd
import re
import ast
import emoji
import json

# Comprehensive Aspect Vocabulary for E-Commerce (Fashion, Beauty, General)
ASPECT_CATEGORIES = {
    "fabric_material": [
        "fabric", "material", "cloth", "kapda", "kapde", "stuff", "silk", "cotton", "khadi", "rayon", "georgette",
        "velvet", "polyester", "wool", "net", "chiffon", "texture"
    ],
    "fit_size": [
        "fitting", "fit", "size", "length", "waist", "loose", "tight", "pant fitting", "kurta fitting", 
        "sleeves", "height", "perfect fit", "dimension"
    ],
    "design_appearance": [
        "design", "look", "looks", "color", "colour", "shade", "print", "embroidery", "work", "pattern",
        "style", "appearance", "shine", "glow", "pigment", "pigmentation", "finish"
    ],
    "quality": [
        "quality", "durability", "build", "stitching", "interlock", "pocket", "pockets", "stiching", 
        "zip", "buttons", "durable", "finish"
    ],
    "price_value": [
        "price", "cost", "paisa", "paise", "rate", "worth", "affordable", "expensive", "sasta", "mehenga",
        "budget", "value for money", "cheap", "overpriced"
    ],
    "delivery_packaging": [
        "delivery", "shipping", "courier", "package", "packaging", "box", "deliver", "packing", 
        "time pe", "late", "damage", "sealed", "safety"
    ],
    "scent_fragrance": [
        "fragrance", "smell", "scent", "khushbu", "khushboo", "odour", "perfume", "aroma"
    ],
    "skin_hair_effect": [
        "skin", "face", "hair", "moisture", "moisturizer", "glow", "softness", "rash", "allergy", 
        "irritation", "hydration", "pimple", "dark spots", "results", "effect"
    ],
    "product_components": [
        "dupatta", "duppatta", "pant", "pants", "kurti", "kurta", "suit", "dress", "saree", "top",
        "shawl", "chunri", "bottom", "palazzo", "plazo", "brush", "applicator", "cream", "lipstick", "serum", "kajal"
    ],
    "seller_service": [
        "seller", "meesho", "service", "customer service", "replacement", "return", "support"
    ]
}

# Lexicons for Hindi/English Code-Mixed ABSA
POSITIVE_TERMS = {
    "good", "nice", "great", "best", "awesome", "super", "superb", "excellent", "amazing", "love", "loved", 
    "perfect", "perfectly", "beautiful", "pyara", "pyari", "mast", "badhiya", "acha", "accha", "achha", "achi", 
    "achhi", "sahi", "fabulous", "wonderful", "satisfied", "satisfactory", "happy", "comfortable", "comfy", 
    "soft", "affordable", "worth", "worthit", "paisa vasool", "paisa wasool", "liked", "like", "recommend", 
    "must buy", "outstanding", "top notch", "flawless", "original", "genuine", "dashing", "cute", "pretty",
    "jabardast", "shandar", "gazab", "kamaal", "lajawab", "mindblowing", "mind blowing"
}

NEGATIVE_TERMS = {
    "bad", "worst", "poor", "terrible", "horrible", "pathetic", "useless", "waste", "bakwas", "bekar", 
    "bekaar", "kharab", "khrab", "bura", "buri", "ghatiya", "fake", "duplicate", "damaged", "damage", 
    "cheap", "rough", "disappointed", "disappointing", "don't buy", "dont buy", "never buy", "not good", 
    "not nice", "faulty", "defective", "loose", "tight", "tight fitting", "chhota", "chhoti", "bada", 
    "badi", "late", "delay", "missing", "faded", "dull", "rash", "allergy", "irritation", "smelly", "bad smell",
    "third class", "waste of money", "paisa barbad", "fraud", "scam"
}

NEGATION_TERMS = {
    "not", "no", "nahi", "nahin", "na", "mat", "never", "without", "bina", "bilkul nahi", "hardly", "barely"
}

INTENSIFIERS = {
    "very", "bahut", "bohot", "bhut", "bhot", "too", "so", "extremely", "really", "bilkul", "kafi", "kaafi", "zyada", "jyada"
}

def clean_review_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Demojize
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_aspects_and_polarity(text, overall_rating=5, mapped_sent="good"):
    """
    Clauses-aware Aspect & Polarity extractor for Hinglish / English E-Commerce reviews.
    """
    clean_t = clean_review_text(text).lower()
    if not clean_t:
        return [], []

    # Tokenize words while preserving punctuation boundaries
    # Split text into clauses on punctuation or coordinating conjunctions (but, lekin, par, aur, and, though)
    clause_delimiters = r"[.,;!?|\n]+|\b(?:but|lekin|magar|par|however|though|although|while|aur|and)\b"
    clauses = [c.strip() for c in re.split(clause_delimiters, clean_t) if c.strip()]
    if not clauses:
        clauses = [clean_t]

    extracted_aspects = []
    extracted_polarities = []

    # Map of all searchable aspect terms
    all_aspect_keywords = {}
    for cat, kws in ASPECT_CATEGORIES.items():
        for kw in kws:
            all_aspect_keywords[kw] = cat

    # Sort keywords by length descending so multi-word terms like "pant fitting" match before "pant"
    sorted_keywords = sorted(all_aspect_keywords.keys(), key=lambda x: len(x), reverse=True)

    found_in_doc = set()

    for clause in clauses:
        clause_words = re.findall(r"[a-z0-9_:]+", clause)
        
        # Check polarity indicators in this clause
        has_negation = any(neg in clause_words for neg in NEGATION_TERMS) or ("nahi" in clause) or ("not" in clause)
        pos_matches = [w for w in clause_words if w in POSITIVE_TERMS or (f":{w}:" in clause and "heart" in w or "thumbs_up" in w or "fire" in w or "star" in w)]
        neg_matches = [w for w in clause_words if w in NEGATIVE_TERMS or (f":{w}:" in clause and "disappointed" in w or "thumbs_down" in w or "rage" in w or "crying" in w)]
        
        # Determine clause polarity
        pos_score = len(pos_matches)
        neg_score = len(neg_matches)
        
        if has_negation:
            # Negation flips positive or reinforces negative
            if pos_score > 0 and neg_score == 0:
                clause_polarity = "negative"
            elif neg_score > 0 and pos_score == 0:
                clause_polarity = "positive" # e.g. "no problem", "bina kisi dikkat"
            else:
                clause_polarity = "negative"
        else:
            if pos_score > neg_score:
                clause_polarity = "positive"
            elif neg_score > pos_score:
                clause_polarity = "negative"
            else:
                # Fallback to overall rating if neutral/unclear
                if overall_rating >= 4:
                    clause_polarity = "positive"
                elif overall_rating <= 2:
                    clause_polarity = "negative"
                else:
                    clause_polarity = "neutral"

        # Find which aspects appear in this clause
        for kw in sorted_keywords:
            # Match whole word boundary
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, clause):
                if kw not in found_in_doc:
                    found_in_doc.add(kw)
                    extracted_aspects.append(kw)
                    extracted_polarities.append(clause_polarity)

    # If no specific aspect was caught, but it's an e-commerce review with strong sentiment, check for general product/quality aspect
    if not extracted_aspects and len(clean_t) > 2:
        if "dress" in clean_t or "suit" in clean_t or "kurti" in clean_t or "product" in clean_t or "item" in clean_t:
            for general_kw in ["product", "dress", "suit", "kurti", "item"]:
                if general_kw in clean_t:
                    extracted_aspects.append(general_kw)
                    default_pol = "positive" if overall_rating >= 4 else ("negative" if overall_rating <= 2 else "neutral")
                    extracted_polarities.append(default_pol)
                    break
        elif any(w in clean_t for w in POSITIVE_TERMS) or any(w in clean_t for w in NEGATIVE_TERMS):
            extracted_aspects.append("product")
            default_pol = "positive" if overall_rating >= 4 else ("negative" if overall_rating <= 2 else "neutral")
            extracted_polarities.append(default_pol)

    return extracted_aspects, extracted_polarities

print("Testing sample reviews annotation...")
samples = [
    ("Fabric bahut achcha hai aur design bhi bahut achcha hai but iske niche mein fitting bekar hai", 3),
    ("Very Good Set. There is quality embroidery work. Also pants have a pocket. Shawl material is not good as expected", 4),
    ("Delivery was very late and packaging was damaged but dress quality is super", 3),
    ("Worst product, cloth quality is pathetic and color faded in first wash. Paisa barbad", 1),
    ("Fragrance is very mild and soothing, skin glows after applying, reasonable price", 5)
]

for s, r in samples:
    asps, pols = extract_aspects_and_polarity(s, overall_rating=r)
    print(f"\nText: {s}")
    print(f"-> Extracted Aspects: {asps}")
    print(f"-> Extracted Polarities: {pols}")
