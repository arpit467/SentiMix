"""
SentiMix ABSA - Next-Generation Enterprise Sentiment Intelligence
Note: Streamlit has been replaced by FastAPI for superior throughput, sub-millisecond API response times, and an executive-grade glassmorphic dashboard.
"""
import os
import sys

if __name__ == "__main__":
    import uvicorn
    import server
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting SentiMix ABSA Production FastAPI Server at http://localhost:{port}")
    uvicorn.run(server.app, host="0.0.0.0", port=port)
