from transformers import AutoTokenizer

models = [
    "ai4bharat/IndicBERTv2-MLM-only",
    "ai4bharat/indic-bert-v2",
    "l3cube-pune/hindi-bert",
    "l3cube-pune/hinglish-bert",
    "bert-base-multilingual-cased"
]

for m in models:
    try:
        t = AutoTokenizer.from_pretrained(m)
        print(f"SUCCESS: {m}")
    except Exception as e:
        print(f"FAILED: {m} -> {e}")
