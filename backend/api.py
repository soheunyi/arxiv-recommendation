#!/usr/bin/env python3
"""
Minimal FastAPI server for ArXiv Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arxiv_recommendation.database import DatabaseManager
    from arxiv_recommendation.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure backend dependencies are installed")

app = FastAPI(
    title="ArXiv Recommendation API",
    description="Backend API for ArXiv Recommendation System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "arxiv-recommendation-api"}

@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    try:
        return {
            "categories": config.arxiv_categories,
            "max_daily_papers": config.max_daily_papers,
            "embedding_model": config.embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers")
async def get_papers(limit: int = 10):
    """Get recent papers"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_recent_papers(limit=limit)
        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
