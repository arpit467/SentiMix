# Aspect-Based Sentiment Analysis (ABSA) for Hindi-English Code-Mixed E-Commerce Reviews (SentiMix)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)

---

## 🎯 1. Project Overview & Research Objectives

Customer reviews on Indian e-commerce platforms (*Meesho, Flipkart, Amazon India*) frequently employ **Hindi-English Code-Mixed (Hinglish)** text with Romanized transliterations (e.g., *"Fabric bahut achha hai but fitting bekar hai"*). Traditional sentence-level sentiment analysis fails to capture multi-aspect polarities where different product features evoke conflicting sentiments.

This project delivers an end-to-end **Aspect-Based Sentiment Analysis (ABSA)** pipeline fulfilling all 5 core research objectives:

1. **Objective 1 (Data Collection & Semi-Supervised Annotation):**
   - Ingested SemEval benchmarks: **Laptop** (3,045 train / 800 test) and **Restaurant** (3,041 train / 800 test).
   - Designed and executed a semi-supervised clause-aware ABSA annotator on the unannotated **Multilingual Meesho Product Review Dataset** (1,005 reviews across Fashion & Beauty).
   - Augmented with a rich **Flipkart/Amazon Hinglish E-Commerce ABSA Dataset** (420 reviews covering Mobile, Electronics, Earphones, and Appliances).
2. **Objective 2 (Code-Mixed Preprocessing Pipeline):**
   - Transliteration spelling canonicalization (e.g., `achha`, `accha`, `acha` $\rightarrow$ `accha`; `bohot`, `bhut` $\rightarrow$ `bahut`; `bekaar` $\rightarrow$ `bekar`).
   - Emoji demojization and sentiment preservation.
   - Code-mixing ratio detection (`is_hinglish`) and clause boundary splitting.
3. **Objective 3 (Aspect Extraction & Categorization):**
   - Fine-grained aspect term & category tagging (*Battery, Camera, Fabric, Fitting, Price, Delivery, Quality, Fragrance, Service, Display, Audio*).
4. **Objective 4 (Multi-Model Sentiment Classification):**
   - **Model 1 (Baseline 1):** Rule-Based Lexicon & Syntactic Proximity Baseline.
   - **Model 2 (Baseline 2):** TF-IDF (Word & Char n-grams) + Multinomial Logistic Regression.
   - **Model 3 (Transformer 1):** IndicBERT (`ai4bharat/IndicBERTv2-MLM-only`).
   - **Model 4 (Main Transformer):** XLM-RoBERTa (`xlm-roberta-base`).
   - **Model 5 (Comparison Transformer):** MuRIL (`google/muril-base-cased`).
5. **Objective 5 (Comparative Benchmarking & Deployment):**
   - Comprehensive multi-metric evaluation (Accuracy, Macro/Weighted F1, Precision, Recall, Hinglish subset F1, and Latency).
   - High-performance **FastAPI Glassmorphic Web Application & REST API** for real-time manual analysis, batch dataset processing with visual charts, and benchmark evaluation.

---

## 🏗️ 2. Repository Architecture

```
NLP/
├── data/
│   ├── complete_laptop_train.csv             # Code-mixed Laptop ABSA Train
│   ├── complete_laptop_test.csv              # Code-mixed Laptop ABSA Test
│   ├── Complete_restaurant_train.csv         # Code-mixed Restaurant ABSA Train
│   ├── Complete_restaurant_test.csv          # Code-mixed Restaurant ABSA Test
│   ├── Multilingual_Meesho_Review_Dataset.csv# Raw Meesho E-commerce reviews
│   ├── meesho_annotated_absa.csv             # Generated High-Quality Meesho ABSA annotations
│   └── augmented_ecommerce_hinglish_absa.csv # Augmented Flipkart/Amazon Hinglish reviews
├── src/
│   ├── torch_fix.py                          # Windows PyTorch DLL initialization
│   ├── preprocessor.py                       # Hinglish normalization, cleaning, demojization
│   ├── annotator.py                          # Automated clause-aware ABSA annotator
│   ├── dataset_loader.py                     # Unified dataset loader & aspect pair generator
│   ├── models/
│   │   ├── rule_based.py                     # Model 1: Rule-Based Baseline
│   │   ├── tfidf_classifier.py               # Model 2: TF-IDF + Logistic Regression
│   │   └── transformer_classifier.py         # Models 3, 4, 5: IndicBERT, XLM-R, MuRIL
│   ├── evaluation.py                         # Evaluation metrics, reports, confusion matrices
│   └── inference_engine.py                   # Unified ABSA inference interface
├── static/
│   ├── index.html                            # Glassmorphic Web Dashboard
│   ├── style.css                             # Dark Modern Design System
│   └── app.js                                # Dynamic Chart.js Analytics & REST API Integration
├── results/
│   ├── saved_models/                         # Trained model weights and checkpoints
│   ├── benchmark_table.csv                   # Comparative performance metrics
│   └── benchmark_comparison.json             # Full evaluation records
├── server.py                                 # FastAPI Server & REST API
├── run_full_pipeline.py                      # Master pipeline training & evaluation script
├── Dockerfile                                # Production Docker container
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## 🚀 3. Quick Start & Execution

### 1. Launch the Modern Web Dashboard & API
```bash
python server.py
```
Open your browser at: **[http://localhost:8000](http://localhost:8000)**

### 2. Run Master Training & Benchmark Pipeline
```bash
python run_full_pipeline.py
```

---

## 📊 4. Models Benchmarked

| Model | Architecture Type | Accuracy | Macro F1 | Hinglish F1 | Latency |
|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | Classical ML (Word+Char N-Grams) | **72.56%** | **64.20%** | **63.62%** | **0.29 ms** |
| **IndicBERT** | Indian Language Transformer (110M) | **63.25%** | 57.23% | 56.94% | 66.70 ms |
| **XLM-RoBERTa** | Cross-Lingual Transformer (278M) | 57.12% | 52.86% | 53.15% | 89.16 ms |
| **MuRIL** | Transliterated Indian BERT (236M) | 40.62% | 31.10% | 29.59% | 63.31 ms |
| **Rule-Based Baseline** | Syntactic Proximity Lexicon | 40.12% | 39.66% | 38.05% | **0.16 ms** |

---

## 🛍️ 5. Dashboard Features

- **⚡ Manual Review Analyzer:** Real-time text analysis with instant aspect extraction, clause context segmentation, and **Polar & Probability distribution charts for the specific review**.
- **📂 Dataset ABSA Studio & Uploader:** Drag-and-drop any CSV/Excel review dataset. Automatically runs high-throughput ABSA and generates **4 dedicated interactive charts for the uploaded dataset**:
  1. *Overall Sentiment Polarity Donut Chart*
  2. *Top Aspect Categories Sentiment Matrix (Stacked Bar)*
  3. *Hinglish Code-Mixing Ratio Pie Chart*
  4. *Top Aspect Terms Frequency Chart*
  Plus an interactive filterable data table and 1-click **Export Enriched ABSA CSV Report**.
- **🏆 5-Model Benchmark Hub:** Interactive comparative performance charts and scientific evaluation tables.
- **📦 Preloaded Datasets Explorer:** Browse 12,604 aspect pairs across Meesho, Flipkart, Amazon, and SemEval.
- **📋 Phase 2 Architecture & Rubric:** Full technical documentation and viva guide.
