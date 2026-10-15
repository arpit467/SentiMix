import os
import sys
import json
import time
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import src.torch_fix
from src.inference_engine import ABSAInferenceEngine
from src.dataset_loader import ABSADatasetLoader

app = FastAPI(
    title="SentiMix ABSA — Enterprise Multilingual Sentiment Intelligence",
    description="Aspect-Based Sentiment Analysis API for Hindi-English Code-Mixed & English E-Commerce Product Reviews",
    version="2.0.0"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize ABSA Engine and Dataset Loader
engine = ABSAInferenceEngine()
loader = ABSADatasetLoader()
cached_datasets = None

def get_datasets():
    global cached_datasets
    if cached_datasets is None:
        try:
            cached_datasets = loader.load_all_datasets()
        except Exception as e:
            print(f"Error loading datasets: {e}")
            cached_datasets = {}
    return cached_datasets

class AnalyzeRequest(BaseModel):
    text: str
    model_name: Optional[str] = "TF-IDF + Logistic Regression"
    custom_aspects: Optional[str] = ""

class BulkAnalyzeRequest(BaseModel):
    reviews: List[str]
    model_name: Optional[str] = "TF-IDF + Logistic Regression"

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
            return HTMLResponse(
                content=content,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    return HTMLResponse("<h1>SentiMix ABSA Dashboard</h1><p>Static index.html not found.</p>")

@app.get("/api/health")
async def health_check():
    return JSONResponse({"status": "healthy", "service": "SentiMix ABSA API", "version": "2.0.0"})

@app.post("/api/analyze")
async def analyze_sentiment(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    t0 = time.time()
    specified = [a.strip() for a in req.custom_aspects.split(",") if a.strip()] if req.custom_aspects else None
    
    result = engine.analyze_review(
        review_text=req.text.strip(),
        model_name=req.model_name,
        specified_aspects=specified
    )
    latency_ms = round((time.time() - t0) * 1000, 2)
    result["latency_ms"] = latency_ms
    return JSONResponse(result)

@app.post("/api/analyze_bulk")
async def analyze_sentiment_bulk(req: BulkAnalyzeRequest):
    valid_reviews = [r.strip() for r in req.reviews if r and str(r).strip()]
    if not valid_reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")
    result = engine.analyze_multiple_reviews(valid_reviews, model_name=req.model_name)
    return JSONResponse(result)

@app.get("/api/benchmarks")
async def get_benchmarks():
    json_path = "results/benchmark_comparison.json"
    table_path = "results/benchmark_table.csv"
    
    table_data = []
    comparison_data = []
    
    if os.path.exists(table_path):
        df = pd.read_csv(table_path)
        table_data = df.to_dict(orient="records")
        
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            comparison_data = json.load(f)
            
    return JSONResponse({
        "table": table_data,
        "details": comparison_data
    })

@app.get("/api/datasets")
async def get_dataset_samples(split: str = "combined_train", limit: int = 50, polarity: str = "all", hinglish_only: bool = False):
    dsets = get_datasets()
    if split not in dsets:
        return JSONResponse({"error": f"Split '{split}' not found", "available_splits": list(dsets.keys())}, status_code=404)
        
    df = dsets[split]
    
    if polarity != "all":
        df = df[df["polarity"] == polarity]
    if hinglish_only:
        df = df[df["is_hinglish"] == True]
        
    total_count = len(df)
    sample_df = df.head(limit)
    
    records = []
    for _, row in sample_df.iterrows():
        records.append({
            "sentence": str(row.get("sentence", "")),
            "aspect": str(row.get("aspect", "")),
            "polarity": str(row.get("polarity", "")),
            "is_hinglish": bool(row.get("is_hinglish", False)),
            "domain": str(row.get("domain", ""))
        })
        
    return JSONResponse({
        "split": split,
        "total_available": total_count,
        "returned_count": len(records),
        "available_splits": list(dsets.keys()),
        "records": records
    })

@app.post("/api/batch")
async def process_batch_file(
    file: UploadFile = File(...),
    model_name: str = Form("TF-IDF + Logistic Regression"),
    text_col: str = Form("")
):
    try:
        t0 = time.time()
        # Read file into DataFrame
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        elif file.filename.endswith(".txt"):
            content = (await file.read()).decode("utf-8", errors="ignore")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if not lines:
                raise HTTPException(status_code=400, detail="Uploaded .txt file is empty.")
            df = pd.DataFrame({"review": lines})
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV, Excel (.xlsx/.xls), or Plain Text (.txt).")
            
        columns = list(df.columns)
        if not columns:
            raise HTTPException(status_code=400, detail="Uploaded file has no columns.")
            
        if not text_col or text_col not in columns:
            # Auto-detect text column
            candidates = [c for c in columns if any(kw in c.lower() for kw in ["comment", "review", "text", "sentence", "feedback", "body", "content", "msg", "message"])]
            text_col = candidates[0] if candidates else columns[0]

        total_reviews = len(df)
        aspect_records = []
        category_counts = {}
        aspect_counts = {}
        overall_sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
        hinglish_count = 0
        total_confidence = 0.0
        aspects_count_total = 0
        enriched_rows = []

        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            if not raw_text.strip():
                continue
                
            res = engine.analyze_review(raw_text, model_name=model_name)
            
            if res["is_hinglish"]:
                hinglish_count += 1
                
            overall = res["overall_sentiment"]
            overall_sentiments[overall] = overall_sentiments.get(overall, 0) + 1
            aspects_in_review = res["aspect_breakdown"]
            aspects_count_total += len(aspects_in_review)
            
            aspect_summary_str = "; ".join([f"{a['aspect']} ({a['sentiment'].upper()})" for a in aspects_in_review])
            
            row_dict = row.to_dict()
            row_dict["Detected_Language"] = "Hinglish" if res["is_hinglish"] else "English"
            row_dict["Overall_Sentiment"] = overall
            row_dict["Aspects_Extracted"] = aspect_summary_str if aspect_summary_str else "Product-level"
            row_dict["Num_Aspects"] = len(aspects_in_review)
            enriched_rows.append(row_dict)
            
            for asp_info in aspects_in_review:
                asp = asp_info["aspect"].lower().strip()
                cat = asp_info.get("category", "general")
                sent = asp_info["sentiment"].lower()
                conf = asp_info.get("confidence", 0.8)
                total_confidence += conf
                
                # Aspect term counts
                if asp not in aspect_counts:
                    aspect_counts[asp] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
                aspect_counts[asp][sent] = aspect_counts[asp].get(sent, 0) + 1
                aspect_counts[asp]["total"] += 1
                
                # Category counts
                cat_clean = cat.replace("_", " ").title()
                if cat_clean not in category_counts:
                    category_counts[cat_clean] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
                category_counts[cat_clean][sent] = category_counts[cat_clean].get(sent, 0) + 1
                category_counts[cat_clean]["total"] += 1
                
                aspect_records.append({
                    "row_index": idx + 1,
                    "review": raw_text[:120] + ("..." if len(raw_text) > 120 else ""),
                    "aspect": asp,
                    "category": cat_clean,
                    "sentiment": sent,
                    "confidence": round(float(conf), 4),
                    "is_hinglish": res["is_hinglish"],
                    "clause": asp_info.get("clause_context", "")
                })

        # Top Categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1]["total"], reverse=True)[:8]
        formatted_top_categories = [
            {"category": k, "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0), "total": v["total"]}
            for k, v in top_categories
        ]

        # Top Aspects
        top_aspect_terms = sorted(aspect_counts.items(), key=lambda x: x[1]["total"], reverse=True)[:10]
        formatted_top_aspects = [
            {"aspect": k, "positive": v.get("positive", 0), "negative": v.get("negative", 0), "neutral": v.get("neutral", 0), "total": v["total"]}
            for k, v in top_aspect_terms
        ]

        avg_conf = round(total_confidence / aspects_count_total * 100, 1) if aspects_count_total > 0 else 85.0
        hinglish_pct = round(hinglish_count / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        pos_pct = round(overall_sentiments.get("Positive", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        neg_pct = round(overall_sentiments.get("Negative", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        neu_pct = round(overall_sentiments.get("Neutral", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        mixed_pct = round(overall_sentiments.get("Mixed", 0) / total_reviews * 100, 1) if total_reviews > 0 else 0.0
        net_sentiment = round(pos_pct - neg_pct, 1)

        # Save enriched CSV
        enriched_df = pd.DataFrame(enriched_rows)
        out_filename = f"absa_analyzed_{int(time.time())}.csv"
        out_path = os.path.join("results", out_filename)
        enriched_df.to_csv(out_path, index=False, encoding="utf-8")

        latency_sec = round(time.time() - t0, 2)

        return JSONResponse({
            "filename": file.filename,
            "total_rows": total_reviews,
            "text_column_used": text_col,
            "available_columns": columns,
            "total_aspects": aspects_count_total,
            "avg_aspects_per_review": round(aspects_count_total / total_reviews, 2) if total_reviews > 0 else 0,
            "hinglish_percentage": hinglish_pct,
            "pure_lang_percentage": round(100.0 - hinglish_pct, 1),
            "avg_confidence": avg_conf,
            "net_sentiment_score": net_sentiment,
            "sentiment_counts": overall_sentiments,
            "sentiment_percentages": {"Positive": pos_pct, "Negative": neg_pct, "Neutral": neu_pct, "Mixed": mixed_pct},
            "top_categories": formatted_top_categories,
            "top_aspect_terms": formatted_top_aspects,
            "download_url": f"/api/download/{out_filename}",
            "aspect_records": aspect_records[:250],
            "sample_results": enriched_rows[:50],
            "latency_sec": latency_sec
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/csv", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[INFO] SentiMix ABSA FastAPI Server running at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
