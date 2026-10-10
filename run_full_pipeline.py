import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import src.torch_fix
import json
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.dataset_loader import ABSADatasetLoader
from src.evaluation import ModelEvaluator
from src.models.rule_based import RuleBasedABSAModel
from src.models.tfidf_classifier import TFIDFLogisticABSAModel
from src.models.transformer_classifier import (
    IndicBERTABSAModel,
    MultilingualMiniLMABSAModel,
    XLMRoBERTaABSAModel,
    MuRILABSAModel
)

def main():
    print("=" * 65)
    print("  ASPECT-BASED SENTIMENT ANALYSIS (ABSA) FOR HINDI-ENGLISH")
    print("  CODE-MIXED E-COMMERCE PRODUCT REVIEWS — MASTER PIPELINE")
    print("=" * 65)
    
    os.makedirs("results/saved_models", exist_ok=True)
    
    # 1. Load Datasets
    print("\n>>> 1. Loading and Preparing Unified Datasets...")
    loader = ABSADatasetLoader(data_dir="data")
    datasets = loader.load_all_datasets()
    
    train_df = datasets["combined_train"]
    test_df = datasets["combined_test"]
    
    print(f"\nUnified Training Set: {len(train_df)} samples")
    print(f"Unified Test Set    : {len(test_df)} samples")
    print(f"Hinglish Test ratio : {test_df['is_hinglish'].mean() * 100:.1f}%")
    
    all_eval_results = []
    
    # -------------------------------------------------------------
    # Model 1: Rule-Based Baseline
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 2. Evaluating Model 1: Rule-Based Baseline")
    print("=" * 50)
    rule_model = RuleBasedABSAModel()
    res_rule = ModelEvaluator.evaluate_model_on_test_set(rule_model, test_df)
    all_eval_results.append(res_rule)
    print(f"Rule-Based -> Accuracy: {res_rule['accuracy']*100:.2f}% | Macro F1: {res_rule['f1_macro']*100:.2f}% | Latency: {res_rule['latency_ms_per_sample']:.2f} ms/item")
    
    # -------------------------------------------------------------
    # Model 2: TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 3. Training & Evaluating Model 2: TF-IDF + Logistic Regression")
    print("=" * 50)
    tfidf_model = TFIDFLogisticABSAModel()
    tfidf_model.train(train_df, val_df=test_df)
    tfidf_save_path = "results/saved_models/tfidf_logistic.joblib"
    tfidf_model.save(tfidf_save_path)
    
    res_tfidf = ModelEvaluator.evaluate_model_on_test_set(tfidf_model, test_df)
    all_eval_results.append(res_tfidf)
    print(f"TF-IDF + LR -> Accuracy: {res_tfidf['accuracy']*100:.2f}% | Macro F1: {res_tfidf['f1_macro']*100:.2f}% | Latency: {res_tfidf['latency_ms_per_sample']:.2f} ms/item")
    
    # -------------------------------------------------------------
    # Model 3: IndicBERT
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 4. Training & Evaluating Model 3: IndicBERT (ai4bharat)")
    print("=" * 50)
    try:
        train_sample = train_df.sample(min(1800, len(train_df)), random_state=42)
        test_sample = test_df.sample(min(800, len(test_df)), random_state=42)
        
        indic_model = IndicBERTABSAModel()
        indic_model.train(train_sample, val_df=test_sample.head(200), batch_size=64)
        indic_save_dir = "results/saved_models/indicbert"
        indic_model.save(indic_save_dir)
        
        res_indic = ModelEvaluator.evaluate_model_on_test_set(indic_model, test_sample, batch_size=64)
        all_eval_results.append(res_indic)
        print(f"IndicBERT -> Accuracy: {res_indic['accuracy']*100:.2f}% | Macro F1: {res_indic['f1_macro']*100:.2f}% | Latency: {res_indic['latency_ms_per_sample']:.2f} ms/item")
    except Exception as e:
        print(f"Error training IndicBERT: {e}")

    # -------------------------------------------------------------
    # Model 4: Multilingual MiniLM (Ultra Fast Cross-Lingual)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 5. Training & Evaluating Model 4: Multilingual MiniLM (sentence-transformers)")
    print("=" * 50)
    try:
        train_sample = train_df.sample(min(1800, len(train_df)), random_state=42)
        test_sample = test_df.sample(min(800, len(test_df)), random_state=42)
        
        minilm_model = MultilingualMiniLMABSAModel()
        minilm_model.train(train_sample, val_df=test_sample.head(200), batch_size=64)
        minilm_save_dir = "results/saved_models/minilm"
        minilm_model.save(minilm_save_dir)
        
        res_minilm = ModelEvaluator.evaluate_model_on_test_set(minilm_model, test_sample, batch_size=64)
        all_eval_results.append(res_minilm)
        print(f"Multilingual MiniLM -> Accuracy: {res_minilm['accuracy']*100:.2f}% | Macro F1: {res_minilm['f1_macro']*100:.2f}% | Latency: {res_minilm['latency_ms_per_sample']:.2f} ms/item")
    except Exception as e:
        print(f"Error training Multilingual MiniLM: {e}")
        
    # -------------------------------------------------------------
    # Model 5: XLM-RoBERTa
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 6. Training & Evaluating Model 5: XLM-RoBERTa")
    print("=" * 50)
    try:
        train_sample = train_df.sample(min(1800, len(train_df)), random_state=42)
        test_sample = test_df.sample(min(800, len(test_df)), random_state=42)
        
        xlm_model = XLMRoBERTaABSAModel()
        xlm_model.train(train_sample, val_df=test_sample.head(200), batch_size=64)
        xlm_save_dir = "results/saved_models/xlm_roberta"
        xlm_model.save(xlm_save_dir)
        
        res_xlm = ModelEvaluator.evaluate_model_on_test_set(xlm_model, test_sample, batch_size=64)
        all_eval_results.append(res_xlm)
        print(f"XLM-RoBERTa -> Accuracy: {res_xlm['accuracy']*100:.2f}% | Macro F1: {res_xlm['f1_macro']*100:.2f}% | Latency: {res_xlm['latency_ms_per_sample']:.2f} ms/item")
    except Exception as e:
        print(f"Error training XLM-RoBERTa: {e}")
        
    # -------------------------------------------------------------
    # Model 6: MuRIL
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> 7. Training & Evaluating Model 6: MuRIL (google/muril-base-cased)")
    print("=" * 50)
    try:
        train_sample = train_df.sample(min(1800, len(train_df)), random_state=42)
        test_sample = test_df.sample(min(800, len(test_df)), random_state=42)
        
        muril_model = MuRILABSAModel()
        muril_model.train(train_sample, val_df=test_sample.head(200), batch_size=64)
        muril_save_dir = "results/saved_models/muril"
        muril_model.save(muril_save_dir)
        
        res_muril = ModelEvaluator.evaluate_model_on_test_set(muril_model, test_sample, batch_size=64)
        all_eval_results.append(res_muril)
        print(f"MuRIL -> Accuracy: {res_muril['accuracy']*100:.2f}% | Macro F1: {res_muril['f1_macro']*100:.2f}% | Latency: {res_muril['latency_ms_per_sample']:.2f} ms/item")
    except Exception as e:
        print(f"Error training MuRIL: {e}")
        
    # -------------------------------------------------------------
    # Comparative Benchmark Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(">>> 8. BENCHMARK COMPARISON TABLE (6 MODELS)")
    print("=" * 65)
    comparison_df = ModelEvaluator.generate_comparison_table(all_eval_results)
    print(comparison_df.to_string(index=False))
    
    # Save results to disk
    table_csv_path = "results/benchmark_table.csv"
    json_path = "results/benchmark_comparison.json"
    comparison_df.to_csv(table_csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2)
        
    print(f"\nAll results saved to:\n- {table_csv_path}\n- {json_path}")
    print("\nPipeline Execution Successfully Completed!")

if __name__ == "__main__":
    main()
