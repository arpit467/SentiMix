import os
import sys
import re
import ast
import nltk
import pandas as pd
from typing import List, Tuple, Dict, Any, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessor import TextPreprocessor, HINGLISH_SPELLING_MAP

# Comprehensive Multi-Domain Aspect Term Dictionary across E-Commerce, Tech, Fashion, Appliances, Beauty, Footwear
# Supporting English, Romanized Hinglish, and Devanagari Hindi
ASPECT_CATEGORIES = {
    # Electronics & Hardware
    "battery_power": [
        "battery", "backup", "charging", "charger", "battery life", "battery backup", "charge",
        "standby", "mah", "drain", "draining", "power", "adapter", "fast charging", "overheating",
        "power consumption", "बिजली", "बिल", "बैटरी", "चार्जिंग", "चार्जर", "पावर"
    ],
    "camera_photography": [
        "camera", "photo", "picture", "pictures", "lens", "selfie", "portrait", "night mode",
        "video", "clarity", "zoom", "sensors", "megapixels", "mp", "shots", "images", "image quality",
        "कैमरा", "फोटो", "पिक्चर", "तस्वीर", "वीडियो"
    ],
    "display_screen": [
        "display", "screen", "amoled", "oled", "refresh rate", "brightness", "resolution",
        "colors", "viewing angles", "panel", "touch", "touchscreen", "120hz", "90hz", "hd",
        "touch response", "रिफ्रेश", "डिस्प्ले", "स्क्रीन", "ब्राइटनेस", "टच"
    ],
    "performance_speed": [
        "performance", "speed", "processor", "ram", "storage", "gaming", "lag", "hanging",
        "smooth", "fps", "heating", "multitasking", "boot time", "ssd", "gpu", "cpu", "thermal",
        "heats", "heat", "hang", "suction", "airflow", "air delivery", "press", "heat up",
        "step counter", "accuracy", "counter", "bluetooth connectivity", "connectivity", "suction power",
        "परफॉरमेंस", "स्पीड", "स्मूद", "काम", "हीटिंग", "गर्म", "सुचारू"
    ],
    "audio_sound": [
        "sound", "audio", "bass", "volume", "speakers", "mic", "microphone", "clarity",
        "noise cancellation", "anc", "earbuds", "headphones", "treble", "voice", "voice recognition",
        "awaaz", "awaz", "shor", "sound quality", "noise", "साउंड", "आवाज़", "माइक", "शोर", "नॉइज़"
    ],
    "connectivity_hardware": [
        "bluetooth", "wifi", "network", "range", "connectivity", "ports", "usb", "keyboard",
        "keypad", "touchpad", "trackpad", "hinge", "buttons", "fingerprint", "sensor", "wire", "cable",
        "ब्लूटूथ", "कनेक्टिविटी", "वायर", "तार", "केबल"
    ],
    # Wearables, Footwear & Comfort
    "comfort_ergonomics": [
        "comfort", "comfortable", "uncomfortable", "strap", "belt", "weight", "lightweight",
        "heavy", "grip", "wear", "fitting", "ergonomic", "softness", "scratch",
        "cushioning", "soft", "chhale", "chhole", "cushion", "heels", "heel", "sole",
        "आराम", "आरामदायक", "आरामदेह", "नरम", "हल्का", "हल्के", "भारी", "पकड़", "ग्रिप", "छाले", "कुशनिंग"
    ],
    # Fashion, Apparel & Accessories
    "fabric_material": [
        "fabric", "material", "cloth", "stuff", "silk", "cotton", "khadi", "rayon",
        "georgette", "velvet", "polyester", "wool", "net", "chiffon", "texture", "kapda", "kapde",
        "denim", "fleece", "leather", "चमड़ा", "कपड़ा", "कपड़े", "फैब्रिक", "सिल्क", "कॉटन", "मटेरियल"
    ],
    "fit_size": [
        "fitting", "fit", "size", "length", "waist", "loose", "tight", "pant fitting",
        "kurta fitting", "sleeves", "height", "perfect fit", "dimension", "shape", "measurement",
        "फिटिंग", "फिट", "साइज़", "नाप", "टाइट", "ढीला", "ढीली", "लंबाई"
    ],
    "design_appearance": [
        "design", "look", "looks", "color", "colour", "shade", "print", "embroidery",
        "work", "pattern", "style", "appearance", "shine", "glow", "finish", "dial",
        "डिज़ाइन", "डिजाइन", "रंग", "शेड", "प्रिंट", "कढ़ाई", "लुक", "स्टाइल", "दिखने"
    ],
    "quality_build": [
        "quality", "durability", "build", "stitching", "interlock", "pocket", "pockets",
        "zip", "buttons", "durable", "finishing", "stitching quality", "body", "frame", "sole",
        "valve", "lid", "jar", "braiding", "tank", "turntable", "elastic", "authenticity",
        "waterproofing", "waterproof", "water resistant", "cleaning", "ease of use", "bottle design",
        "cord", "ink", "print quality", "water tastes", "water quality", "accessories", "attachments",
        "क्वालिटी", "गुणवत्ता", "सिलाई", "सोल", "मजबूती", "टिकाऊ", "बिल्ड", "लीकेज", "लीक", "ढक्कन"
    ],
    # Commercial & Services
    "price_value": [
        "price", "cost", "paisa", "paise", "rate", "worth", "affordable", "expensive",
        "sasta", "mehenga", "budget", "value for money", "cheap", "overpriced", "deal", "discount",
        "कीमत", "दाम", "पैसे", "पैसा", "सस्ता", "सस्ते", "महंगा", "महंगी", "रेट", "वैल्यू"
    ],
    "delivery_service": [
        "delivery", "shipping", "courier", "package", "packaging", "box", "deliver",
        "packing", "time pe", "late", "delayed", "damage", "sealed", "safety", "seller", "meesho", "flipkart", "amazon",
        "service", "customer support", "support", "customer care", "customer service", "installation", "return", "return process", "logistics",
        "डिलीवरी", "पैकेजिंग", "पैकिंग", "सर्विस", "कस्टमर केयर", "कस्टमर सपोर्ट", "डिलीवर", "इंस्टालेशन"
    ],
    # Beauty & Personal Care
    "scent_fragrance": [
        "fragrance", "smell", "scent", "khushbu", "khushboo", "odour", "perfume", "aroma",
        "खुशबू", "सुगंध", "बदबू", "स्मेल", "महक"
    ],
    "skin_beauty_effect": [
        "skin", "face", "hair", "moisture", "moisturizer", "glow", "softness", "rash",
        "allergy", "irritation", "hydration", "pimple", "pimples", "dark spots", "results", "effect",
        "breakouts", "ammonia", "sunscreen", "white cast", "coverage", "shade match", "smudge resistance",
        "effectiveness", "smudge proof", "longevity", "protection", "brush", "applicator",
        "त्वचा", "चेहरा", "बाल", "ग्लो", "रैश", "मुहांसे", "पिंपल्स", "रिजल्ट", "असर"
    ],
    "cooling_temperature": [
        "cooling", "cool", "heating", "warmth", "water tastes", "water quality", "कूलिंग", "ठंडक", "गर्मी", "हीटिंग", "हवा"
    ],

    # Product Categories
    "product_item": [
        "phone", "mobile", "laptop", "earphones", "earbud", "earbuds", "watch", "smartwatch", "tv", "camera",
        "suit", "dress", "kurti", "kurta", "dupatta", "pant", "pants", "shawl", "chunri", "chappal", "sandals", "slides",
        "saree", "top", "palazzo", "cream", "lipstick", "serum", "kajal", "product", "item", "set", "device", "crocs",
        "fan", "cooker", "ac", "air conditioner", "fridge", "refrigerator", "blender", "iron", "heater", "geyser", "soundbar",
        "shoes", "sneakers", "boots", "loafers", "heels", "frock", "hoodie", "jacket", "toaster", "kettle",
        "सूट", "साड़ी", "कुर्ती", "लहंगा", "दुपट्टा", "केतली", "फ्रिज", "वॉशिंग मशीन", "फैन", "पंखा", "जैकेट", "हुडी", "जूते", "चप्पल"
    ]
}

# Positive sentiment lexicons (English + Hinglish + Devanagari)
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

# Negative sentiment lexicons (English + Hinglish + Devanagari)
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

# Neutral lexicons
NEUTRAL_LEXICON = {
    "theek theek", "theek", "average", "normal", "nothing special", "okay", "okay okay",
    "okay type", "chalta hai", "theek thaak", "gets the job done", "औसत", "सामान्य", "ठीक-ठाक",
    "ठीक ठाक", "न बुरा न अच्छा", "neither great nor bad", "neither great or bad", "does the job", "expected", "theek hai"
}


NEGATION_TOKENS = {
    "not", "no", "nahi", "nahin", "na", "mat", "never", "without", "bina", "hardly", "barely",
    "don't", "dont", "नहीं", "ना", "न", "मत", "बिना"
}

# Sarcastic cues indicating overall dissatisfaction despite positive surface words (Generalized without hardcoded brands)
SARCASM_PATTERNS = [
    r"\bnot\b[\s\.!]*$",
    r"\bthanks for the (?:free|gift|damaged|broken|pimples)\b",
    r"\b(?:wah|shabash|kya baat hai|great job|well done)\b.*(?:hang|leak|chhale|barbad|bekar|kharab|phati|mazaa|dead|worst|pathetic|useless)",
    r"\bवाह[,\s]+क्या\b.*(?:खराब|टूटी|उधड़|बर्बाद|छाले|क्वालिटी|कचरा)",
    r"\bkya baat hai.*service ho toh aisi\b",
    r"\bbrilliant quality control, well done\b"
]


# Strict Stop Words & Intensifiers (MUST NEVER be treated as aspect terms)
STOP_WORDS_AND_MODIFIERS = {
    "is", "was", "are", "were", "the", "a", "an", "and", "or", "to", "in", "on", "for", "with",
    "hai", "hain", "tha", "thi", "the", "ho", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "bhi", "bahut", "bohot", "bhut", "bhot", "very", "kaafi", "kafi", "kfi", "thoda",
    "thodi", "thora", "zyada", "jada", "jyada", "bilkul", "ekdum", "sirf", "toh", "to", "so",
    "too", "really", "quite", "ab", "jab", "tab", "kuch", "khas", "laga", "lagi", "lage",
    "lagta", "lagti", "lagte", "raha", "rahi", "rahe", "hona", "hone", "hota", "hoti", "hote",
    "karo", "karna", "kare", "wali", "wala", "wale", "yeh", "ye", "woh", "wo", "mera", "meri",
    "mere", "apna", "apni", "apne", "after", "before", "while", "then", "just", "also", "even",
    "but", "par", "lekin", "magar", "bhai", "yaar", "plz", "please", "all", "stopped", "impressive",
    "fantastic", "purchase", "truly", "doctor", "lena", "mahine", "days", "day", "first", "work",
    "works", "stays", "though", "though.", "looks", "used", "wearing", "using", "buy", "buying",
    "takes", "taking", "made", "came", "ordered", "runs", "getting", "got", "even",
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "से", "में", "भी", "बहुत", "काफी",
    "थोड़ा", "थोड़ी", "ज्यादा", "बिल्कुल", "एकदम", "तो", "अब", "जब", "तब", "कुछ", "खास", "लगा",
    "लगी", "रहा", "रही", "रहे", "होना", "होता", "होती", "करो", "करना", "करे", "वाले", "वाला",
    "वाली", "यह", "ये", "वह", "वो", "मेरा", "मेरी", "मेरे", "अपना", "अपनी", "अपने", "लेकिन",
    "पर", "मगर", "और", "या", "बस", "कुल", "मिलाकर", "हो", "गया", "गई", "गए", "पड़ेगा", "पड़ेगी"
}


class DynamicAspectExtractor:
    """
    Completely Dynamic Open-Vocabulary Aspect Extractor:
    1. Segments reviews into clauses around contrastive markers.
    2. Runs NLTK POS-tagging + Noun Phrase (NP) chunking (<JJ|JJR|JJS>* <NN|NNS|NNP|NNPS>+)
       to discover arbitrary, unseen product aspects and compound terms from any customer review.
    3. Extracts Romanized code-mixed Hindi (Hinglish) and Devanagari Unicode nouns.
    4. Filters out stopwords, pure sentiment adjectives, and grammatical particles.
    5. Dynamically assigns aspect categories using semantic keyword clustering or head noun semantics.
    """
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.grammar = r"""
            NP: {<JJ|JJR|JJS>* <NN|NNS|NNP|NNPS>+}
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        
        # Build flattened keyword lookup sorted by length descending
        self.keyword_to_category = {}
        for cat, kws in ASPECT_CATEGORIES.items():
            for kw in kws:
                self.keyword_to_category[kw.lower()] = cat
        self.sorted_keywords = sorted(self.keyword_to_category.keys(), key=lambda x: len(x), reverse=True)
        self.compiled_aspect_patterns = [
            (kw, self.keyword_to_category[kw.lower()], re.compile(r'(?<![a-zA-Z0-9\u0900-\u097F])' + re.escape(kw.lower()) + r'(?![a-zA-Z0-9\u0900-\u097F])'))
            for kw in self.sorted_keywords
        ]
        
        # Stop words & sentiment words that should never be aspect terms
        self.ignore_tokens = set(STOP_WORDS_AND_MODIFIERS) | set(POSITIVE_LEXICON) | set(NEGATIVE_LEXICON) | set(NEUTRAL_LEXICON) | set(NEGATION_TOKENS)
        self.ignore_tokens.update({
            "is", "was", "are", "were", "the", "a", "an", "and", "or", "to", "in", "on", "for", "with",
            "days", "day", "night", "month", "months", "time", "product", "item", "aur", "mila", "mile",
            "total", "vibrant", "soft", "loud", "noisy", "fast", "slow", "heavy", "good", "bad", "stars",
            "star", "reviews", "review", "order", "ordered"
        })
        
        self.devanagari_ignore = {
            "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "से", "में", "भी", "बहुत", "काफी",
            "थोड़ा", "थोड़ी", "ज्यादा", "बिल्कुल", "एकदम", "तो", "अब", "जब", "तब", "कुछ", "खास", "लगा",
            "लगी", "रहा", "रही", "रहे", "होना", "होता", "होती", "करो", "करना", "करे", "वाले", "वाला",
            "वाली", "यह", "ये", "वह", "वो", "मेरा", "मेरी", "मेरे", "अपना", "अपनी", "अपने", "लेकिन",
            "पर", "मगर", "और", "या", "बस", "कुल", "मिलाकर", "हो", "गया", "गई", "गए", "पड़ेगा", "पड़ेगी",
            "अच्छा", "अच्छी", "अच्छे", "बढ़िया", "शानदार", "खराब", "घटिया", "बेकार", "बर्बाद", "बुरा",
            "छोटा", "बड़ा", "सस्ता", "महंगा", "सही", "गलत", "मिला", "मिली", "मिले", "दिया", "दी", "गया"
        }

    def infer_category(self, aspect: str) -> str:
        asp = aspect.lower().strip()
        # Direct lookup
        if asp in self.keyword_to_category:
            return self.keyword_to_category[asp]
            
        # Substring lookup in taxonomy
        for kw, cat, pat in self.compiled_aspect_patterns:
            if pat.search(asp):
                return cat
                
        # Semantic dynamic fallbacks
        if any(w in asp for w in ["battery", "backup", "charging", "charger", "power", "बैटरी", "चार्ज"]):
            return "battery_power"
        if any(w in asp for w in ["camera", "photo", "picture", "lens", "selfie", "zoom", "कैमरा", "तस्वीर"]):
            return "camera_photography"
        if any(w in asp for w in ["display", "screen", "amoled", "oled", "120hz", "90hz", "brightness", "डिस्प्ले", "स्क्रीन"]):
            return "display_screen"
        if any(w in asp for w in ["speed", "performance", "ram", "processor", "lag", "fps", "heating", "boot", "परफॉरमेंस", "स्पीड"]):
            return "performance_speed"
        if any(w in asp for w in ["sound", "audio", "bass", "speaker", "speakers", "mic", "volume", "voice", "awaaz", "awaz", "साउंड", "आवाज़"]):
            return "audio_sound"
        if any(w in asp for w in ["fabric", "cloth", "material", "silk", "cotton", "stuff", "kapda", "कपड़ा", "कपड़े", "सिल्क"]):
            return "fabric_material"
        if any(w in asp for w in ["fit", "fitting", "size", "length", "waist", "सूट", "फिटिंग", "साइज़"]):
            return "fit_size"
        if any(w in asp for w in ["design", "look", "looks", "color", "colour", "shade", "print", "स्टाइल", "रंग"]):
            return "design_appearance"
        if any(w in asp for w in ["delivery", "courier", "shipping", "package", "packaging", "box", "agent", "seller", "support", "refund", "policy", "service", "customer", "डिलीवरी", "पैकिंग"]):
            return "delivery_service"
        if any(w in asp for w in ["price", "cost", "paisa", "rate", "worth", "affordable", "expensive", "sasta", "दाम", "पैसे", "कीमत"]):
            return "price_value"
        if any(w in asp for w in ["comfort", "comfortable", "strap", "cushion", "sole", "आराम"]):
            return "comfort_ergonomics"
        if any(w in asp for w in ["quality", "build", "stitching", "durable", "body", "frame", "zipper", "क्वालिटी"]):
            return "quality_build"
        if any(w in asp for w in ["fragrance", "smell", "scent", "khushbu", "खुशबू"]):
            return "scent_fragrance"
        if any(w in asp for w in ["skin", "face", "hair", "moisture", "glow", "त्वचा", "चेहरा"]):
            return "skin_beauty_effect"
        return "product_feature"

    def extract_aspects(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Dynamically extracts aspect terms, their inferred categories, and clause contexts.
        Returns: [(aspect_term, category, clause_context)]
        """
        if not text or not str(text).strip():
            return []
            
        clauses = self.preprocessor.split_into_clauses(text)
        candidates = []
        
        for clause in clauses:
            cl_clean = clause.strip()
            if not cl_clean:
                continue
                
            cl_lower = cl_clean.lower()
            
            # Step A: Check for established multi-word and single-word taxonomy terms
            for kw, cat, pat in self.compiled_aspect_patterns:
                if pat.search(cl_lower):
                    kw_clean = kw.strip().lower()
                    if kw_clean not in self.ignore_tokens:
                        candidates.append((kw_clean, cat, cl_clean))
            
            # Step B: Devanagari Hindi Dynamic Aspect Discovery
            dev_words = re.findall(r"[\u0900-\u097F]+", cl_clean)
            for dw in dev_words:
                if dw not in self.devanagari_ignore and len(dw) > 1:
                    cat = self.infer_category(dw)
                    candidates.append((dw.lower(), cat, cl_clean))
                    
            # Step C: English / Romanized Hinglish POS Noun Phrase Chunking (Captures unseen novel terms)
            tokens = nltk.word_tokenize(cl_clean)
            if tokens:
                try:
                    tagged = nltk.pos_tag(tokens)
                    tree = self.chunker.parse(tagged)
                    
                    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
                        phrase_tokens = [w.lower() for w, _ in subtree.leaves()]
                        valid_words = [
                            w for w in phrase_tokens
                            if w not in self.ignore_tokens
                            and len(w) > 2
                            and not w.isdigit()
                            and not bool(re.search(r"[\u0900-\u097F]", w))
                        ]
                        
                        if valid_words:
                            aspect_term = " ".join(valid_words)
                            aspect_term = re.sub(r"^[^\w]+|[^\w]+$", "", aspect_term).strip()
                            if aspect_term and len(aspect_term) > 2:
                                cat = self.infer_category(aspect_term)
                                candidates.append((aspect_term, cat, cl_clean))
                except Exception:
                    pass
                    
        # Step D: Maximal Phrase Subsumption & Deduplication (keep specific compound phrases over redundant fragments)
        candidates = sorted(candidates, key=lambda x: len(x[0]), reverse=True)
        dedup_results = []
        seen_aspects = set()
        
        for asp, cat, cl in candidates:
            if asp in seen_aspects:
                continue
            # Check if this aspect is a redundant sub-word of an already kept aspect in the same clause
            is_subsumed = False
            for k_asp, k_cat, k_cl in dedup_results:
                if cl == k_cl and asp in k_asp and asp != k_asp:
                    is_subsumed = True
                    break
            if not is_subsumed:
                seen_aspects.add(asp)
                dedup_results.append((asp, cat, cl))
                
        # Step E: Fallback if no specific noun was detected
        if not dedup_results:
            dedup_results.append(("product", "product_item", text))
            
        return dedup_results


class MeeshoABSAAnnotator:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.dynamic_extractor = DynamicAspectExtractor()
        self.keyword_to_category = self.dynamic_extractor.keyword_to_category

    def _determine_clause_sentiment(self, clause: str, overall_rating: float = 4.0) -> str:
        clause_clean = clause.lower()
        words = set(re.findall(r"[a-z0-9_:]+|[\u0900-\u097F]+", clause_clean))
        
        # Sarcasm check
        for spat in SARCASM_PATTERNS:
            if re.search(spat, clause_clean, re.IGNORECASE):
                return "negative"
                
        # Check Neutral indicators
        for neu in NEUTRAL_LEXICON:
            if neu in clause_clean:
                return "neutral"
        
        emoji_pos = any(e in clause_clean for e in [":thumbs_up:", ":heart:", ":sparkles:", ":fire:", ":smiling_", ":star:", ":folded_hands:"])
        emoji_neg = any(e in clause_clean for e in [":thumbs_down:", ":crying_", ":rage:", ":disappointed:", ":broken_heart:", ":weary:"])
        
        has_negation = any(neg in words for neg in NEGATION_TOKENS) or ("nahi" in clause_clean) or ("not" in clause_clean) or ("mat " in clause_clean) or ("नहीं" in clause_clean)
        
        # Direct phrase matches
        pos_phrases = ["not bad", "paisa vasool", "paisa wasool", "value for money", "पैसे वसूल", "must buy", "definitely buy", "no issues", "no problem", "zero damage", "not drying at all", "no white cast", "not see-through"]
        neg_phrases = ["waste of money", "paisa barbad", "stopped working", "bilkul bekaar", "dead on arrival", "पैसे बर्बाद", "मत खरीदना", "मत लेना", "fake product", "worst quality", "leaky", "headache", "pockets bade hain par fabric", "total waste"]
        
        for p in pos_phrases:
            if p in clause_clean:
                return "positive"
        for p in neg_phrases:
            if p in clause_clean:
                return "negative"
                
        pos_count = sum(1 for w in words if w in POSITIVE_LEXICON) + (1 if emoji_pos else 0)
        neg_count = sum(1 for w in words if w in NEGATIVE_LEXICON) + (1 if emoji_neg else 0)
        
        if has_negation:
            if pos_count > 0 and neg_count == 0:
                if "not drying" in clause_clean or "no dryness" in clause_clean:
                    return "positive"
                return "negative"
            elif neg_count > 0:
                if any(good_neg in clause_clean for good_neg in ["no pain", "no foot pain", "no white cast", "no complaints", "no issues", "zero damage", "no problem", "awaaz bilkul nahi", "awaaz nahi", "noise bilkul nahi", "no dryness"]):
                    return "positive"
                return "negative"
            else:
                return "negative"
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            if "theek" in clause_clean or "average" in clause_clean or "normal" in clause_clean or "okay" in clause_clean or "औसत" in clause_clean or "सामान्य" in clause_clean:
                return "neutral"
            if overall_rating >= 4.0:
                return "positive"
            elif overall_rating <= 2.0:
                return "negative"
            else:
                return "neutral"

    def annotate_review(self, text: str, rating: float = 4.0, mapped_sentiment: str = "") -> Dict[str, Any]:
        clean_norm = self.preprocessor.normalize_hinglish(text)
        
        aspect_terms = []
        at_polarity = []
        aspect_categories = []
        
        # Sarcasm check across whole text
        is_sarcastic = any(re.search(p, clean_norm.lower()) for p in SARCASM_PATTERNS)
        
        # Dynamically extract all aspects and clauses
        extracted_triples = self.dynamic_extractor.extract_aspects(text)
        
        for asp, cat, cl_context in extracted_triples:
            clause_sent = self._determine_clause_sentiment(cl_context, rating)
            if is_sarcastic:
                clause_sent = "negative"
            aspect_terms.append(asp)
            at_polarity.append(clause_sent)
            aspect_categories.append(cat)
            
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
