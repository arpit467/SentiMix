import re

clause = "fabric bahut achcha hai"
clean_t = clause.lower()
clause_words = re.findall(r"[a-z0-9_:]+", clean_t)
POSITIVE_TERMS = {
    "good", "nice", "great", "best", "awesome", "super", "superb", "excellent", "amazing", "love", "loved", 
    "perfect", "perfectly", "beautiful", "pyara", "pyari", "mast", "badhiya", "acha", "accha", "achha", "achi", 
    "achhi", "sahi", "fabulous", "wonderful", "satisfied", "satisfactory", "happy", "comfortable", "comfy", 
    "soft", "affordable", "worth", "worthit", "paisa vasool", "paisa wasool", "liked", "like", "recommend", 
    "must buy", "outstanding", "top notch", "flawless", "original", "genuine", "dashing", "cute", "pretty",
    "jabardast", "shandar", "gazab", "kamaal", "lajawab", "mindblowing", "mind blowing"
}
for w in clause_words:
    print(f"word '{w}' in POSITIVE_TERMS: {w in POSITIVE_TERMS}")
