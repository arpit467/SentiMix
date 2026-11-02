import os
import sys
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import src.torch_fix
from src.dataset_loader import ABSADatasetLoader
from src.models.transformer_classifier import MultilingualMiniLMABSAModel, IndicBERTABSAModel
from src.evaluation import ModelEvaluator

def create_balanced_subset(df: pd.DataFrame, n_per_class: int = 150) -> pd.DataFrame:
    """Creates a balanced subset across all 3 classes (negative, neutral, positive)."""
    subsets = []
    for label_id in [0, 1, 2]:
        class_df = df[df["label"] == label_id]
        n_take = min(n_per_class, len(class_df))
        subsets.append(class_df.sample(n_take, random_state=42))
    balanced_df = pd.concat(subsets).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return balanced_df

import argparse

def train_and_eval_model(model_cls, model_name, save_dir, train_sub, test_sub, epochs, batch_size, lr=2e-5):
    print("\n" + "=" * 60)
    print(f">>> Fine-Tuning {model_name}...")
    print("=" * 60)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model_cls()
    
    t0 = time.time()
    ft_metrics = model.fine_tune(
        train_df=train_sub,
        val_df=test_sub.head(min(200, len(test_sub))),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        checkpoint_dir=save_dir,
        use_amp=True
    )
    ft_time = time.time() - t0
    print(f"Fine-tuning completed in {ft_time:.1f} seconds ({ft_time/60:.2f} mins)!")
    sys.stdout.flush()

    # Save fine-tuned checkpoint
    print(f"\nSaving fine-tuned model checkpoint to {save_dir}...")
    model.save(save_dir)

    # Evaluate fine-tuned model on test benchmark
    print(f"\nEvaluating fine-tuned {model_name} on test set...")
    res_ft = ModelEvaluator.evaluate_model_on_test_set(model, test_sub, batch_size=batch_size)
    
    print("\n" + "=" * 60)
    print(f"  FINE-TUNED {model_name.upper()} TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy         : {res_ft['accuracy']*100:.2f}%")
    print(f"Macro F1         : {res_ft['f1_macro']*100:.2f}%")
    print(f"Weighted F1      : {res_ft['f1_weighted']*100:.2f}%")
    print(f"Hinglish Subset  : Accuracy: {res_ft.get('hinglish_subset', {}).get('accuracy', 0.0)*100:.2f}% | F1: {res_ft.get('hinglish_subset', {}).get('f1_macro', 0.0)*100:.2f}%")
    print(f"Latency / Item   : {res_ft['latency_ms_per_sample']:.2f} ms")
    print("=" * 60)

    # Live test on edge-case Hinglish reviews
    print(f"\n>>> Testing Live Inference on fine-tuned {model_name}:")
    test_reviews = [
        ("Screen accha hai but battery bekar hai", "screen"),
        ("Screen accha hai but battery bekar hai", "battery"),
        ("Fabric bahut soft hai fitting tight hai", "fabric"),
        ("Fabric bahut soft hai fitting tight hai", "fitting"),
        ("Camera mast hai but pricing bahut zyada hai", "camera"),
        ("Camera mast hai but pricing bahut zyada hai", "pricing")
    ]
    for sent, asp in test_reviews:
        pol, pid, probs = model.predict_pair(sent, asp)
        print(f"Review: '{sent}' | Aspect: '{asp}' -> Sentiment: {pol.upper()} (conf: {probs[pol]*100:.1f}%)")
    return model, res_ft

def main():
    parser = argparse.ArgumentParser(description="End-to-End Fine-Tuning for Transformer ABSA Models")
    parser.add_argument("--model", type=str, default="minilm", choices=["minilm", "indicbert", "both"],
                        help="Which model to fine-tune (default: minilm)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 3 on GPU, 2 on CPU)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: 16)")
    args = parser.parse_args()

    print("=" * 65)
    print("  SENTIMIX ABSA: TRUE END-TO-END TRANSFORMER FINE-TUNING")
    print("=" * 65)

    loader = ABSADatasetLoader(data_dir="data")
    datasets = loader.load_all_datasets()
    
    train_df = datasets["combined_train"]
    test_df = datasets["combined_test"]
    
    print(f"Master Train Set: {len(train_df):,} aspect pairs")
    print(f"Master Test Set : {len(test_df):,} aspect pairs")

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {gpu_name} (CUDA). Using full training set for fine-tuning!")
        train_sub = train_df
        test_sub = test_df
        epochs = args.epochs or 3
        batch_size = args.batch_size or 32  # 32 with FP16 & max_length=64 runs 4x faster on GTX 1650 (~6 mins total)
    else:
        print("Running on CPU. Creating balanced training subset (450 samples: 150 Pos, 150 Neg, 150 Neu)...")
        train_sub = create_balanced_subset(train_df, n_per_class=150)
        test_sub = test_df.sample(min(400, len(test_df)), random_state=42)
        epochs = args.epochs or 2
        batch_size = args.batch_size or 16

    print(f"Selected {len(train_sub)} samples for fine-tuning across classes: {dict(train_sub['label'].value_counts())}")

    if args.model in ["minilm", "both"]:
        train_and_eval_model(
            model_cls=MultilingualMiniLMABSAModel,
            model_name="Multilingual MiniLM",
            save_dir="results/saved_models/minilm",
            train_sub=train_sub,
            test_sub=test_sub,
            epochs=epochs,
            batch_size=batch_size,
            lr=2e-5
        )

    if args.model in ["indicbert", "both"]:
        train_and_eval_model(
            model_cls=IndicBERTABSAModel,
            model_name="IndicBERT",
            save_dir="results/saved_models/indicbert",
            train_sub=train_sub,
            test_sub=test_sub,
            epochs=epochs,
            batch_size=batch_size,
            lr=2e-5
        )

if __name__ == "__main__":
    main()

