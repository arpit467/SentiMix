import re

clause = "fabric bahut achcha hai"
clean_t = clause.lower()
clause_words = re.findall(r"[a-z0-9_:]+", clean_t)
print("clause words:", clause_words)
POSITIVE_TERMS = {"good", "nice", "great", "best", "awesome", "super", "superb", "excellent", "amazing", "love", "loved", "perfect", "perfectly", "beautiful", "pyara", "pyari", "mast", "badhiya", "acha", "accha", "achha", "achi", "achhi", "sahi"}
pos_matches = [w for w in clause_words if w in POSITIVE_TERMS]
print("pos matches:", pos_matches)
