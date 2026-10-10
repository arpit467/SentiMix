import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("PyTorch Version:", torch.__version__)
print("Checking model architectures...")

models_to_test = [
    ("IndicBERT", "ai4bharat/indic-bert"),
    ("XLM-RoBERTa", "xlm-roberta-base"),
    ("MuRIL", "google/muril-base-cased")
]

for name, model_id in models_to_test:
    print(f"\n--- Testing {name} ({model_id}) ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer for {name} loaded successfully!")
        sample_inputs = tokenizer("यह product bahut accha hai", "product", return_tensors="pt")
        print("Sample tokenized input keys:", sample_inputs.keys())
    except Exception as e:
        print(f"Error loading {name}: {e}")
