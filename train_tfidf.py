import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import src.torch_fix
import pandas as pd
from src.dataset_loader import ABSADatasetLoader
from src.models.tfidf_classifier import TFIDFLogisticABSAModel
from src.inference_engine import ABSAInferenceEngine

def main():
    loader = ABSADatasetLoader()
    datasets = loader.load_all_datasets()
    train_df = datasets["combined_train"]
    test_df = datasets["combined_test"]
    
    print(f"Retraining TF-IDF on {len(train_df)} samples...")
    model = TFIDFLogisticABSAModel()
    model.train(train_df, val_df=test_df)
    model.save("results/saved_models/tfidf_logistic.joblib")
    
    print("\nTesting ABSA Inference Engine on diverse test inputs:")
    engine = ABSAInferenceEngine()
    
    test_cases = [
        "Display bahut vibrant hai 120Hz AMOLED aur battery life 2 days chalti hai lekin camera zoom night me bekar hai",
        "Fabric is super soft and fitting is perfect but colour faded after one wash",
        "Ekdum ghatiya product hai mat khareedo",
        "Delivery was late and packing was torn but headphone sound clarity is awesome",
        "Screen is very bright and keyboard is smooth",
        "Not bad quality for this price, paisa vasool",
        "Worst battery backup, lasts only 1 hour and heats up rapidly",
        "Camera quality is average but processor speed is ultra fast"
    ]
    
    for tc in test_cases:
        res = engine.analyze_review(tc)
        print(f"\n--- Input: {tc} ---")
        print(f"Overall: {res['overall_sentiment']}")
        for asp in res["aspect_breakdown"]:
            print(f"  -> Aspect: '{asp['aspect']}' [{asp['category']}] => Sentiment: {asp['sentiment'].upper()} (Confidence: {asp['confidence']*100:.1f}%)")

if __name__ == "__main__":
    main()
