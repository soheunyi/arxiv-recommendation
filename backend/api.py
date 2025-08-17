#!/usr/bin/env python3
"""
Minimal FastAPI server for ArXiv Recommendation System
"""

import asyncio
import time
import uuid
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import traceback
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Enhanced structured logging setup
class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'user_agent'):
            log_entry["user_agent"] = record.user_agent
        if hasattr(record, 'ip_address'):
            log_entry["ip_address"] = record.ip_address
        if hasattr(record, 'endpoint'):
            log_entry["endpoint"] = record.endpoint
        if hasattr(record, 'method'):
            log_entry["method"] = record.method
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
        if hasattr(record, 'response_time'):
            log_entry["response_time_ms"] = record.response_time
        
        return json.dumps(log_entry)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set up structured formatter for our logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(StructuredFormatter())
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Also set up a separate access logger
access_logger = logging.getLogger("api.access")
access_handler = logging.StreamHandler(sys.stdout)
access_handler.setFormatter(StructuredFormatter())
access_logger.handlers.clear()
access_logger.addHandler(access_handler)
access_logger.setLevel(logging.INFO)

try:
    from database import DatabaseManager
    from config import config
    from services.backup_service import BackupService

    logger.info("Successfully imported ArXiv recommendation modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure backend dependencies are installed")
    raise

app = FastAPI(
    title="ArXiv Recommendation API",
    description="Backend API for ArXiv Recommendation System",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str
    scope: Optional[str] = "all"
    dateRange: Optional[str] = "all"
    categories: Optional[list] = []
    sortBy: Optional[str] = "relevance"
    page: Optional[int] = 1
    limit: Optional[int] = 20


class RatingRequest(BaseModel):
    paper_id: str
    rating: int
    notes: Optional[str] = None


class CollectionRequest(BaseModel):
    keyword: str
    max_papers: Optional[int] = 20
    clean_db: Optional[bool] = False
    cleanup_options: Optional[Dict] = None
    llm_provider: Optional[str] = None  # "openai" or "gemini"


class ManualPaperRequest(BaseModel):
    arxiv_id: str
    category: Optional[str] = None  # Optional category override


def create_response(
    data: Any, success: bool = True, message: str = None
) -> Dict[str, Any]:
    """Create standardized API response"""
    return {
        "data": data,
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


def create_error_response(error: Exception, message: str = None, correlation_id: str = None) -> Dict[str, Any]:
    """Create standardized error response with detailed logging"""
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    error_msg = message or str(error)
    error_type = type(error).__name__

    # Enhanced error logging with structured data
    logger.error(
        f"Error {error_id}: {error_msg}",
        extra={
            "error_id": error_id,
            "error_type": error_type,
            "correlation_id": correlation_id,
        },
        exc_info=True
    )

    return {
        "data": None,
        "success": False,
        "message": error_msg,
        "error_id": error_id,
        "error_type": error_type,
        "timestamp": datetime.now().isoformat(),
    }


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Comprehensive logging and error handling middleware"""
    # Generate correlation ID for request tracking
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    
    # Extract request info
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    start_time = time.time()
    
    # Log incoming request
    access_logger.info(
        "Incoming request",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "endpoint": str(request.url),
            "ip_address": client_ip,
            "user_agent": user_agent,
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # Log successful response
        access_logger.info(
            "Request completed",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "endpoint": str(request.url),
                "status_code": response.status_code,
                "response_time": response_time,
                "ip_address": client_ip,
            }
        )
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{response_time}ms"
        
        return response
        
    except Exception as e:
        # Calculate response time for error case
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # Log error with full context
        logger.error(
            f"Unhandled exception in {request.method} {request.url}: {str(e)}",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "endpoint": str(request.url),
                "ip_address": client_ip,
                "user_agent": user_agent,
                "response_time": response_time,
                "exception_type": type(e).__name__,
            },
            exc_info=True
        )
        
        # Create error response with correlation ID
        error_response = create_error_response(e, "Internal server error")
        error_response["correlation_id"] = correlation_id
        
        # Create JSON response with correlation ID header
        json_response = JSONResponse(status_code=500, content=error_response)
        json_response.headers["X-Correlation-ID"] = correlation_id
        json_response.headers["X-Response-Time"] = f"{response_time}ms"
        
        return json_response


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database and perform startup checks"""
    try:
        logger.info("Starting ArXiv Recommendation API...")

        # Test database connection
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("Database initialized successfully")

        # Test configuration
        if not config.openai_api_key:
            logger.warning("OpenAI API key not configured - embeddings will not work")

        # Initialize scheduler
        try:
            from scheduler import start_scheduler

            await start_scheduler()
            logger.info("Task scheduler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {e}")
            # Don't fail startup if scheduler fails

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        logger.info("Shutting down ArXiv Recommendation API...")

        # Stop scheduler
        try:
            from scheduler import stop_scheduler

            await stop_scheduler()
            logger.info("Task scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")

        logger.info("API shutdown completed")

    except Exception as e:
        logger.error(f"Shutdown failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return create_response({"status": "healthy", "service": "arxiv-recommendation-api"})


@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    try:
        logger.info("Fetching system configuration")
        data = {
            "categories": config.arxiv_categories,
            "max_daily_papers": config.max_daily_papers,
            "embedding_model": config.embedding_model,
        }
        return create_response(data)
    except AttributeError as e:
        logger.error(f"Configuration attribute missing: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch configuration: {str(e)}"
        )


@app.get("/api/papers")
async def get_papers(
    page: int = 1,
    limit: int = 20,
    category: str = None,
    rating: str = None,
    search: str = None,
    start_date: str = None,
    end_date: str = None,
):
    """Get paginated papers with filters"""
    try:
        logger.info(f"Fetching papers: page={page}, limit={limit}, category={category}")

        if page < 1:
            raise ValueError("Page number must be >= 1")
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")

        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Calculate offset for pagination
        offset = (page - 1) * limit

        # For now, return recent papers - can be enhanced later with filtering
        papers = await db_manager.get_recent_papers(limit=limit + offset)

        # Apply pagination manually for now
        paginated_papers = (
            papers[offset : offset + limit] if len(papers) > offset else []
        )

        pagination_data = {
            "items": paginated_papers,
            "total": len(papers),
            "page": page,
            "pageSize": limit,
            "totalPages": (len(papers) + limit - 1) // limit,
            "hasNext": offset + limit < len(papers),
            "hasPrevious": page > 1,
        }
        return create_response(pagination_data)
    except ValueError as e:
        logger.error(f"Invalid pagination parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch papers: {str(e)}")


@app.get("/api/papers/all")
async def get_all_papers():
    """Get all papers"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_papers_with_ratings()
        return create_response(papers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/recent")
async def get_recent_papers(limit: int = 20, sort_by: str = "created_at"):
    """Get recent papers with optional sorting"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        if sort_by == "score":
            # Get papers sorted by recommendation score
            papers = await db_manager.get_recent_recommendations(limit=limit)
        else:
            # Default: get recent papers sorted by creation date
            papers = await db_manager.get_recent_papers(limit=limit)

        return create_response(papers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/for-rating")
async def get_papers_for_rating(
    category: str = None, rating_filter: str = None, limit: int = 50
):
    """Get papers for rating interface"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_papers_with_ratings()

        # Apply basic filtering
        if category and category != "All":
            papers = [p for p in papers if p.get("category") == category]

        if rating_filter and rating_filter != "All":
            if rating_filter == "unrated":
                papers = [
                    p for p in papers if not p.get("rating") or p.get("rating") == 0
                ]
            else:
                rating_val = int(rating_filter)
                papers = [p for p in papers if p.get("rating") == rating_val]

        return create_response(papers[:limit])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/categories")
async def get_categories():
    """Get available paper categories"""
    try:
        categories = config.arxiv_categories
        return create_response(categories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/stats")
async def get_paper_stats():
    """Get enhanced paper statistics with backup-relevant counts"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_papers_with_ratings()

        # Calculate basic stats
        total = len(papers)
        by_category = {}
        total_rating = 0
        rated_count = 0

        for paper in papers:
            category = paper.get("category", "Unknown")
            by_category[category] = by_category.get(category, 0) + 1

            rating = paper.get("rating", 0)
            if rating > 0:
                total_rating += rating
                rated_count += 1

        avg_rating = total_rating / rated_count if rated_count > 0 else 0

        # Get detailed counts for backup purposes
        import aiosqlite

        async with aiosqlite.connect(db_manager.db_path) as db:
            # Count records in each table for cleanup modal
            counts = {}
            tables = [
                "papers",
                "user_ratings",
                "paper_embeddings",
                "search_history",
                "recommendations_history",
            ]

            for table in tables:
                try:
                    cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                    count = await cursor.fetchone()
                    counts[f"{table}_count"] = count[0] if count else 0
                except Exception:
                    counts[f"{table}_count"] = 0

        stats = {
            "total": total,
            "by_category": by_category,
            "recent_additions": len([p for p in papers[:10]]),  # Last 10 as recent
            "avg_rating": round(avg_rating, 2),
            **counts,  # Add individual table counts
        }
        return create_response(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get comprehensive analytics overview including paper count and rating distribution"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_papers_with_ratings()

        # Basic counts
        total_papers = len(papers)
        rated_papers = len([p for p in papers if p.get("rating", 0) > 0])
        unrated_papers = total_papers - rated_papers

        # Rating distribution (1-5 stars)
        rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        total_rating_sum = 0

        for paper in papers:
            rating = paper.get("rating", 0)
            if rating > 0:
                rating_distribution[rating] += 1
                total_rating_sum += rating

        # Category distribution
        category_stats = {}
        category_ratings = {}

        for paper in papers:
            category = paper.get("category", "Unknown")
            rating = paper.get("rating", 0)

            if category not in category_stats:
                category_stats[category] = {"total": 0, "rated": 0, "rating_sum": 0}

            category_stats[category]["total"] += 1
            if rating > 0:
                category_stats[category]["rated"] += 1
                category_stats[category]["rating_sum"] += rating

        # Calculate averages and percentages for categories
        for category in category_stats:
            stats = category_stats[category]
            avg_rating = (
                stats["rating_sum"] / stats["rated"] if stats["rated"] > 0 else 0
            )
            percentage = (
                (stats["total"] / total_papers * 100) if total_papers > 0 else 0
            )

            category_ratings[category] = {
                "total_papers": stats["total"],
                "rated_papers": stats["rated"],
                "average_rating": round(avg_rating, 2),
                "percentage": round(percentage, 1),
            }

        # Overall statistics
        overall_avg_rating = total_rating_sum / rated_papers if rated_papers > 0 else 0
        completion_rate = (rated_papers / total_papers * 100) if total_papers > 0 else 0

        analytics = {
            "overview": {
                "total_papers": total_papers,
                "rated_papers": rated_papers,
                "unrated_papers": unrated_papers,
                "completion_rate": round(completion_rate, 1),
                "average_rating": round(overall_avg_rating, 2),
            },
            "rating_distribution": rating_distribution,
            "category_breakdown": category_ratings,
            "summary": {
                "most_common_rating": (
                    max(rating_distribution.items(), key=lambda x: x[1])[0]
                    if any(rating_distribution.values())
                    else None
                ),
                "total_categories": len(category_ratings),
                "best_rated_category": (
                    max(category_ratings.items(), key=lambda x: x[1]["average_rating"])[
                        0
                    ]
                    if category_ratings
                    else None
                ),
            },
        }

        return create_response(analytics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/generate-sample-scores")
async def generate_sample_scores():
    """Generate sample recommendation scores for testing purposes"""
    try:
        import random

        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Get all papers
        papers = await db_manager.get_papers_with_ratings()

        # Generate sample recommendations with scores
        recommendations = []
        for paper in papers[:50]:  # Limit to first 50 papers
            score = round(random.uniform(0.1, 0.9), 3)
            recommendations.append({"paper_id": paper["id"], "score": score})

        # Store the recommendations (this will update current_score)
        if recommendations:
            await db_manager.store_recommendations(recommendations, "sample_v1.0")

        return create_response(
            {
                "generated_scores": len(recommendations),
                "message": f"Generated sample scores for {len(recommendations)} papers",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get single paper by ID (supports both ROWID and ArXiv ID)"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Check if paper_id is numeric (ROWID) or string (ArXiv ID)
        if paper_id.isdigit():
            # Use ROWID lookup
            paper = await db_manager.get_paper_by_rowid(int(paper_id))
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
        else:
            # Use ArXiv ID lookup
            papers = await db_manager.get_papers_with_ratings()
            paper = next((p for p in papers if p.get("id") == paper_id), None)
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")

        return create_response(paper)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/search")
async def search_papers(search_request: SearchRequest):
    """Search papers with filters"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_papers_with_ratings()

        # Simple text search implementation
        query = search_request.query.lower()
        filtered_papers = []

        for paper in papers:
            # Search in title, abstract, and authors
            searchable_text = ""
            if search_request.scope == "all" or search_request.scope == "title":
                searchable_text += paper.get("title", "").lower()
            if search_request.scope == "all" or search_request.scope == "abstract":
                searchable_text += " " + paper.get("abstract", "").lower()
            if search_request.scope == "all" or search_request.scope == "authors":
                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    searchable_text += " " + " ".join(authors).lower()
                else:
                    searchable_text += " " + str(authors).lower()

            if query in searchable_text:
                filtered_papers.append(paper)

        # Apply category filter
        if search_request.categories:
            filtered_papers = [
                p
                for p in filtered_papers
                if p.get("category") in search_request.categories
            ]

        # Simple sorting
        if search_request.sortBy == "date":
            filtered_papers.sort(
                key=lambda x: x.get("published_date", ""), reverse=True
            )
        elif search_request.sortBy == "title":
            filtered_papers.sort(key=lambda x: x.get("title", ""))

        # Pagination
        start_idx = (search_request.page - 1) * search_request.limit
        end_idx = start_idx + search_request.limit
        paginated_papers = filtered_papers[start_idx:end_idx]

        result = {
            "papers": paginated_papers,
            "total": len(filtered_papers),
            "hasMore": end_idx < len(filtered_papers),
            "facets": {"categories": {}, "ratings": {}},
        }

        return create_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/papers/manual-add")
async def add_manual_paper(request: ManualPaperRequest):
    """Manually add an ArXiv paper by ID."""
    try:
        import re
        from arxiv_client import ArXivClient
        
        # Validate ArXiv ID format
        arxiv_id = request.arxiv_id.strip()
        if not re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid ArXiv ID format. Expected format: YYYY.NNNNN or YYYY.NNNNNvN, got: {arxiv_id}"
            )
        
        # Check if paper already exists
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        existing_paper = await db_manager.get_paper_by_id(arxiv_id)
        if existing_paper:
            return create_response(
                {"paper": existing_paper, "already_exists": True},
                message=f"Paper {arxiv_id} already exists in database"
            )
        
        # Fetch paper from ArXiv
        logger.info(f"📝 Manual paper entry: Fetching {arxiv_id} from ArXiv")
        
        async with ArXivClient() as client:
            paper = await client.get_paper_by_id(arxiv_id)
            
        if not paper:
            raise HTTPException(
                status_code=404,
                detail=f"Paper {arxiv_id} not found on ArXiv"
            )
        
        # Override category if provided
        if request.category:
            paper.category = request.category
            logger.info(f"📝 Overriding category for {arxiv_id}: {request.category}")
        
        # Store paper in database (pass PaperMetadata object directly)
        success = await db_manager.store_papers([paper])
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store paper {arxiv_id} in database"
            )
        
        logger.info(f"✅ Successfully added manual paper: {arxiv_id}")
        
        # Create response with paper data as dict for API response
        paper_data = {
            "id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "category": paper.category,
            "published_date": paper.published_date.isoformat(),
            "updated_date": paper.updated_date.isoformat(),
            "arxiv_url": paper.arxiv_url,
            "pdf_url": paper.pdf_url,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "source": "manual",  # Track manual entry
            "added_manually": True
        }
        
        return create_response(
            {
                "paper": paper_data,
                "manually_added": True,
                "arxiv_id": arxiv_id
            },
            message=f"Successfully added paper {arxiv_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Manual paper entry failed for {request.arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Recommendation endpoints
@app.get("/api/recommendations/recent")
async def get_recommendations(limit: int = 10):
    """Get recent recommendations"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # For now, return recent papers as recommendations
        papers = await db_manager.get_recent_papers(limit=limit)

        # Format as recommendation result
        result = {
            "recommendations": papers,
            "total": len(papers),
            "generated_at": datetime.now().isoformat(),
            "algorithm": "recent_papers_fallback",
            "parameters": {"limit": limit},
        }

        return create_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommendations/generate")
async def generate_recommendations():
    """Generate new recommendations"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # For now, return recent papers as recommendations
        papers = await db_manager.get_recent_papers(limit=10)

        # Format as recommendation result
        result = {
            "recommendations": papers,
            "total": len(papers),
            "generated_at": datetime.now().isoformat(),
            "algorithm": "recent_papers_fallback",
            "parameters": {"user_based": False, "content_based": False},
        }

        return create_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ratings/user")
async def get_user_ratings():
    """Get all user ratings"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Get all user ratings from the database
        papers = await db_manager.get_papers_with_ratings()
        ratings = []

        for paper in papers:
            if paper.get("rating") and paper.get("rating") > 0:
                rating_data = {
                    "id": f"rating_{paper['id']}",  # Generate a unique ID
                    "paper_id": paper["id"],
                    "rating": paper["rating"],
                    "notes": paper.get("notes", ""),
                    "created_at": paper.get("created_at", ""),
                    "updated_at": paper.get("updated_at", paper.get("created_at", "")),
                }
                ratings.append(rating_data)

        return create_response(ratings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ratings/update")
async def update_rating(rating_request: RatingRequest):
    """Update or create a paper rating"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Update the rating in the database
        success = await db_manager.store_user_rating(
            paper_id=rating_request.paper_id,
            rating=rating_request.rating,
            notes=rating_request.notes or "",
        )

        if success:
            result = {
                "paper_id": rating_request.paper_id,
                "rating": rating_request.rating,
                "notes": rating_request.notes,
            }
            return create_response(result)
        else:
            raise HTTPException(status_code=400, detail="Failed to update rating")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Reference tracking endpoints
@app.get("/api/papers/{paper_id}/references")
async def get_paper_references(paper_id: str):
    """Get all references for a paper (supports both ROWID and ArXiv ID)."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Convert ROWID to ArXiv ID if needed (references are stored by ArXiv ID)
        arxiv_id = paper_id
        if paper_id.isdigit():
            # Get paper by ROWID to find the ArXiv ID
            paper = await db_manager.get_paper_by_rowid(int(paper_id))
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            arxiv_id = paper.get("id")  # ArXiv ID is stored in the 'id' field
        
        references = await db_manager.get_paper_references(arxiv_id)
        reference_count = await db_manager.get_reference_count(arxiv_id)
        
        result = {
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "references": references,
            "reference_count": reference_count
        }
        
        return create_response(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching references for paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/{paper_id}/citations")
async def get_paper_citations(paper_id: str):
    """Get all papers that cite this paper."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        citations = await db_manager.get_paper_citations(paper_id)
        citation_count = await db_manager.get_citation_count(paper_id)
        
        result = {
            "paper_id": paper_id,
            "citations": citations,
            "citation_count": citation_count
        }
        
        return create_response(result)
    except Exception as e:
        logger.error(f"Error fetching citations for paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/references/fetch/{paper_id}")
async def fetch_paper_references(paper_id: str, force_refresh: bool = False):
    """
    Trigger hybrid reference fetching for a specific paper (supports both ROWID and ArXiv ID).
    Uses two-stage approach: ArXiv HTML parsing + OpenAlex enrichment.
    """
    try:
        import asyncio
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Handle both ROWID and ArXiv ID
        if paper_id.isdigit():
            # Use ROWID lookup
            paper = await db_manager.get_paper_by_rowid(int(paper_id))
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            arxiv_id = paper.get("id")  # ArXiv ID is stored in the 'id' field
        else:
            # Use ArXiv ID lookup
            papers = await db_manager.get_papers_with_ratings()
            paper = next((p for p in papers if p.get("id") == paper_id), None)
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            arxiv_id = paper_id
        
        # Validate ArXiv ID extraction
        if not arxiv_id:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract arXiv ID from paper"
            )
        
        # Use GROBID service for PDF-based reference extraction
        from services.grobid_service import GrobidService
        
        async with GrobidService() as grobid_service:
            # Extract references using GROBID
            references = await grobid_service.extract_references_from_arxiv(arxiv_id)
            
            # Store references in database
            if references:
                await db_manager.store_references(arxiv_id, references)
            
            # Create result in expected format
            fetch_result = {
                "arxiv_id": arxiv_id,
                "references_found": len(references),
                "stage1_success": len(references) > 0,
                "stage2_success": False,  # No second stage with GROBID-only approach
                "source": "grobid",
                "openalex_available": False,  # OpenAlex removed
            }
        
        # Return comprehensive result from hybrid service
        response_data = {
            "paper_id": paper_id,
            "arxiv_id": fetch_result["arxiv_id"],
            "references_found": fetch_result["references_found"],
            "stage1_success": fetch_result["stage1_success"],
            "stage2_success": fetch_result["stage2_success"],
            "source": fetch_result["source"],
            "openalex_available": fetch_result["openalex_available"],
            "status": "completed",
            "timestamp": fetch_result["timestamp"]
        }
        
        if "error" in fetch_result:
            response_data["error"] = fetch_result["error"]
        
        return create_response(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching references for paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/references/network/{paper_id}")
async def get_citation_network(paper_id: str, depth: int = 1):
    """Get citation network for visualization (optional enhancement)."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Start with the target paper
        network = {"nodes": [], "edges": []}
        processed_papers = set()
        
        async def add_paper_to_network(pid: str, level: int = 0):
            if pid in processed_papers or level > depth:
                return
                
            processed_papers.add(pid)
            
            # Get paper info
            papers = await db_manager.get_papers_with_ratings()
            paper = next((p for p in papers if p.get("id") == pid), None)
            
            if paper:
                network["nodes"].append({
                    "id": pid,
                    "title": paper.get("title", ""),
                    "category": paper.get("category", ""),
                    "published_date": paper.get("published_date", ""),
                    "level": level
                })
                
                # Add references (outgoing edges)
                references = await db_manager.get_paper_references(pid)
                for ref in references:
                    if ref.get("cited_paper_id"):
                        network["edges"].append({
                            "source": pid,
                            "target": ref["cited_paper_id"],
                            "type": "references"
                        })
                        await add_paper_to_network(ref["cited_paper_id"], level + 1)
                
                # Add citations (incoming edges)
                citations = await db_manager.get_paper_citations(pid)
                for cit in citations:
                    network["edges"].append({
                        "source": cit["citing_paper_id"],
                        "target": pid,
                        "type": "cites"
                    })
                    await add_paper_to_network(cit["citing_paper_id"], level + 1)
        
        await add_paper_to_network(paper_id)
        
        result = {
            "paper_id": paper_id,
            "network": network,
            "stats": {
                "nodes": len(network["nodes"]),
                "edges": len(network["edges"]),
                "depth": depth
            }
        }
        
        return create_response(result)
        
    except Exception as e:
        logger.error(f"Error building citation network for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collection/start")
async def start_collection(collection_request: CollectionRequest):
    """Start a new paper collection in background."""
    try:
        import uuid
        import asyncio
        from pathlib import Path

        collection_id = str(uuid.uuid4())

        # Store collection in database
        db_manager = DatabaseManager()
        await db_manager.initialize()

        await db_manager.create_collection(
            collection_id,
            collection_request.keyword,
            {
                "max_papers": collection_request.max_papers,
                "clean_db": collection_request.clean_db,
            },
        )

        # Start background collection
        asyncio.create_task(run_collection_task(collection_id, collection_request))

        return create_response(
            {
                "collection_id": collection_id,
                "status": "started",
                "keyword": collection_request.keyword,
                "estimated_time": 60,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collection/status/{collection_id}")
async def get_collection_status(collection_id: str):
    """Get collection progress status."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        status = await db_manager.get_collection_status(collection_id)
        if not status:
            raise HTTPException(status_code=404, detail="Collection not found")

        return create_response(status)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collection/preview")
async def preview_collection(collection_request: CollectionRequest):
    """Preview what would be collected without executing."""
    try:
        from services import CollectionService

        collector = CollectionService(collection_request.keyword)
        await collector.generate_queries(
            max_queries=collection_request.max_papers or 15,
            llm_provider=collection_request.llm_provider
        )

        preview = collector.preview_collection_plan()
        return create_response(preview)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collection/generate-queries")
async def generate_queries(request: dict):
    """Generate search queries for a topic with optional date filtering."""
    try:
        from services.provider_factory import ProviderFactory

        topic = request.get("topic")
        max_queries = request.get("max_queries", 15)
        llm_provider = request.get("llm_provider")
        date_from = request.get("date_from")  # YYYY-MM-DD format
        date_to = request.get("date_to")      # YYYY-MM-DD format

        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")

        # Validate date format if provided
        if date_from:
            try:
                from datetime import datetime
                datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="date_from must be in YYYY-MM-DD format"
                )
        
        if date_to:
            try:
                from datetime import datetime
                datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="date_to must be in YYYY-MM-DD format"
                )

        query_service = ProviderFactory.create_query_service(llm_provider)
        queries = query_service.generate_search_queries(
            topic, 
            max_queries, 
            date_from, 
            date_to
        )

        return create_response(queries)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collection/providers")
async def get_llm_providers():
    """Get available LLM providers and their information."""
    try:
        from services.provider_factory import ProviderFactory
        from config import config
        
        comparison = ProviderFactory.compare_providers()
        current_provider = config.llm_provider
        
        return create_response({
            "current_provider": current_provider,
            "providers": comparison,
            "recommendation": ProviderFactory.recommend_provider(
                cost_sensitive=True,
                multilingual=False, 
                long_context=False
            )
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# Collaborative Service Endpoints
# ======================

@app.post("/api/collaboration/generate-queries")
async def generate_queries_collaborative(request: dict):
    """Generate queries using collaborative OpenAI-Gemini approach."""
    try:
        from services.collaborative_service import collaborative_service, CollaborationStrategy

        topic = request.get("topic")
        max_queries = request.get("max_queries", 15)
        date_from = request.get("date_from")
        date_to = request.get("date_to")
        strategy = request.get("strategy", "adaptive")
        quality_threshold = request.get("quality_threshold", 0.8)

        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required")

        # Validate strategy
        try:
            strategy_enum = CollaborationStrategy(strategy)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy. Must be one of: {[s.value for s in CollaborationStrategy]}"
            )

        # Validate date format if provided
        if date_from:
            try:
                from datetime import datetime
                datetime.strptime(date_from, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="date_from must be in YYYY-MM-DD format"
                )
        
        if date_to:
            try:
                from datetime import datetime
                datetime.strptime(date_to, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="date_to must be in YYYY-MM-DD format"
                )

        result = await collaborative_service.generate_queries_collaborative(
            topic=topic,
            max_queries=max_queries,
            date_from=date_from,
            date_to=date_to,
            strategy=strategy_enum,
            quality_threshold=quality_threshold
        )

        # Convert to response format
        response_data = {
            "queries": result.primary_result,
            "collaboration_info": {
                "primary_provider": result.primary_provider,
                "secondary_provider": result.secondary_provider,
                "strategy_used": result.strategy_used,
                "quality_score": result.quality_score,
                "cost_estimate": result.cost_estimate,
                "execution_time": result.execution_time,
                "has_secondary_result": result.secondary_result is not None
            }
        }

        if result.secondary_result:
            response_data["alternative_queries"] = result.secondary_result

        return create_response(response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collaborative query generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collaboration/strategies")
async def get_collaboration_strategies():
    """Get available collaboration strategies and their descriptions."""
    from services.collaborative_service import CollaborationStrategy
    
    strategies = {
        CollaborationStrategy.COST_OPTIMIZED.value: {
            "name": "Cost Optimized",
            "description": "Use Gemini primarily for cost savings, OpenAI for quality validation",
            "use_case": "Budget-conscious scenarios with quality assurance",
            "cost_impact": "Low",
            "quality_impact": "High"
        },
        CollaborationStrategy.QUALITY_FIRST.value: {
            "name": "Quality First",
            "description": "Use OpenAI primarily, Gemini as backup",
            "use_case": "Critical applications requiring highest quality",
            "cost_impact": "High",
            "quality_impact": "Highest"
        },
        CollaborationStrategy.PARALLEL_COMPARE.value: {
            "name": "Parallel Compare",
            "description": "Run both providers in parallel and select best result",
            "use_case": "Complex topics requiring comprehensive coverage",
            "cost_impact": "Highest",
            "quality_impact": "Highest"
        },
        CollaborationStrategy.ADAPTIVE.value: {
            "name": "Adaptive",
            "description": "Automatically select best strategy based on context",
            "use_case": "General purpose with intelligent optimization",
            "cost_impact": "Variable",
            "quality_impact": "High"
        }
    }
    
    return create_response({
        "strategies": strategies,
        "default": CollaborationStrategy.ADAPTIVE.value,
        "recommendation": "Use adaptive strategy for optimal balance of cost and quality"
    })


@app.get("/api/collaboration/usage-stats")
async def get_collaboration_usage_stats():
    """Get current usage statistics for collaborative service."""
    try:
        from services.collaborative_service import collaborative_service
        
        stats = collaborative_service.get_usage_stats()
        
        return create_response(stats)
        
    except Exception as e:
        logger.error(f"Failed to get usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collaboration/switch-provider")
async def switch_default_provider(request: dict):
    """Switch the default LLM provider."""
    try:
        from services.provider_factory import switch_provider
        
        provider = request.get("provider")
        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        
        success = switch_provider(provider)
        if not success:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
        
        return create_response({
            "switched": True,
            "new_provider": provider,
            "note": "This change is temporary for the current session. Update environment variable for persistence."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch provider: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/collection/save-config")
async def save_collection_config(request: dict):
    """Save collection configuration to file."""
    try:
        from services import QueryService

        config_data = request.get("config")
        filepath = request.get("filepath")

        if not config_data or not filepath:
            raise HTTPException(
                status_code=400, detail="Config and filepath are required"
            )

        query_service = QueryService()
        success = query_service.save_queries_config(config_data, filepath)

        if success:
            return create_response({"saved": True, "filepath": filepath})
        else:
            raise HTTPException(status_code=500, detail="Failed to save configuration")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collection/load-config")
async def load_collection_config(filepath: str):
    """Load collection configuration from file."""
    try:
        from services import QueryService

        query_service = QueryService()
        config_data = query_service.load_queries_config(filepath)

        if config_data:
            return create_response(config_data)
        else:
            raise HTTPException(status_code=404, detail="Configuration file not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backup endpoints
@app.post("/api/backup/create")
async def create_backup(request: dict):
    """Create a backup of specified data."""
    try:
        from services import BackupService

        backup_type = request.get("backup_type", "ratings")

        backup_service = BackupService()
        result = await backup_service.create_backup(backup_type)

        return create_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup/list")
async def list_backups():
    """List all available backups."""
    try:
        from services import BackupService

        backup_service = BackupService()
        backups = backup_service.list_backups()

        return create_response(backups)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backup/restore")
async def restore_backup(request: dict):
    """Restore from a backup file."""
    try:
        from services import BackupService

        backup_filename = request.get("backup_filename")
        backup_type = request.get("backup_type", "ratings")

        if not backup_filename:
            raise HTTPException(status_code=400, detail="Backup filename is required")

        backup_service = BackupService()

        if backup_type == "ratings":
            result = await backup_service.restore_ratings(backup_filename)
        elif backup_type == "full":
            result = await backup_service.restore_full_database(backup_filename)
        else:
            raise HTTPException(status_code=400, detail="Invalid backup type")

        return create_response(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/backup/{backup_filename}")
async def delete_backup(backup_filename: str):
    """Delete a backup file."""
    try:
        from services import BackupService

        backup_service = BackupService()
        result = backup_service.delete_backup(backup_filename)

        return create_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Scheduler Management Endpoints


@app.get("/api/scheduler/status")
async def get_scheduler_status():
    """Get status of all scheduled tasks."""
    try:
        from scheduler import get_scheduler

        scheduler = get_scheduler()
        status = scheduler.get_task_status()

        return create_response(status)

    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get scheduler status: {str(e)}"
        )


@app.post("/api/scheduler/trigger/{task_name}")
async def trigger_scheduled_task(task_name: str):
    """Manually trigger a specific scheduled task."""
    try:
        from scheduler import get_scheduler

        valid_tasks = [
            "daily_collection",
            "daily_scoring",
            "preference_update",
            "cache_maintenance",
        ]
        if task_name not in valid_tasks:
            raise HTTPException(
                status_code=400, detail=f"Invalid task name. Valid tasks: {valid_tasks}"
            )

        scheduler = get_scheduler()
        result = await scheduler.trigger_task(task_name)

        return create_response(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering task {task_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger task: {str(e)}")


# Preference Management Endpoints


class PreferenceModeRequest(BaseModel):
    mode: str  # "all_time", "recent", or "adaptive"


@app.get("/api/preferences/mode")
async def get_preference_mode():
    """Get the current user preference mode."""
    try:
        from preferences import PreferenceManager

        preference_manager = PreferenceManager()
        current_mode = await preference_manager.get_preference_mode()

        return create_response({"mode": current_mode.value})

    except Exception as e:
        logger.error(f"Error getting preference mode: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get preference mode: {str(e)}"
        )


@app.put("/api/preferences/mode")
async def set_preference_mode(request: PreferenceModeRequest):
    """Set the user preference mode."""
    try:
        from preferences import PreferenceManager, PreferenceMode

        # Validate mode
        valid_modes = [mode.value for mode in PreferenceMode]
        if request.mode not in valid_modes:
            raise HTTPException(
                status_code=400, detail=f"Invalid mode. Valid modes: {valid_modes}"
            )

        # Convert string to enum
        mode_enum = None
        for mode in PreferenceMode:
            if mode.value == request.mode:
                mode_enum = mode
                break

        preference_manager = PreferenceManager()
        await preference_manager.set_preference_mode(mode_enum)

        return create_response({"mode": request.mode, "status": "updated"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting preference mode: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to set preference mode: {str(e)}"
        )


@app.get("/api/preferences/stats")
async def get_preference_stats():
    """Get preference management statistics."""
    try:
        from preferences import PreferenceManager

        preference_manager = PreferenceManager()
        stats = preference_manager.get_preference_stats()

        return create_response(stats)

    except Exception as e:
        logger.error(f"Error getting preference stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get preference stats: {str(e)}"
        )


# Scoring Service Endpoints


@app.get("/api/scoring/stats")
async def get_scoring_stats():
    """Get scoring service statistics."""
    try:
        from scoring_service import ScoringService

        scoring_service = ScoringService()
        stats = scoring_service.get_scoring_stats()

        return create_response(stats)

    except Exception as e:
        logger.error(f"Error getting scoring stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get scoring stats: {str(e)}"
        )


@app.post("/api/scoring/update")
async def trigger_scoring_update():
    """Manually trigger a scoring update."""
    try:
        from scoring_service import ScoringService

        scoring_service = ScoringService()
        result = await scoring_service.update_daily_scores()

        return create_response(
            {
                "papers_scored": result.papers_scored,
                "papers_skipped": result.papers_skipped,
                "processing_time": result.processing_time_seconds,
                "cache_hits": result.embedding_cache_hits,
                "cache_misses": result.embedding_cache_misses,
                "errors": result.errors,
            }
        )

    except Exception as e:
        logger.error(f"Error updating scores: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update scores: {str(e)}"
        )


async def run_collection_task(collection_id: str, request: CollectionRequest):
    """Background task to run paper collection."""
    try:
        from services import CollectionService

        db_manager = DatabaseManager()
        await db_manager.initialize()

        # Initialize collector
        collector = CollectionService(request.keyword)

        # Update status to running
        await db_manager.update_collection_status(collection_id, "running", progress=0)

        # Generate queries with specified provider
        await collector.generate_queries(max_queries=10, llm_provider=request.llm_provider)
        total_queries = len(collector.query_config["search_queries"])

        await db_manager.update_collection_status(
            collection_id,
            "running",
            progress=10,
            total_queries=total_queries,
            current_query="Generating search queries...",
        )

        # Enhanced database cleaning with backup and granular options
        if request.clean_db:
            cleanup_options = getattr(request, "cleanup_options", None)
            backup_metadata = {
                "keyword": request.keyword,
                "max_papers": request.max_papers,
                "user_initiated": True,
                "collection_id": collection_id,
            }

            cleanup_result = await collector.clean_database(
                backup_metadata=backup_metadata, granular_options=cleanup_options
            )

            if not cleanup_result["success"]:
                await db_manager.update_collection_status(
                    collection_id,
                    "failed",
                    progress=0,
                    current_query=f"Database cleanup failed: {cleanup_result.get('error', 'Unknown error')}",
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Database cleanup failed: {cleanup_result.get('error')}",
                )

            logger.info(
                f"Database cleanup completed: {cleanup_result['total_records_removed']} records removed"
            )
            logger.info(f"Backup created: {cleanup_result['backup_info']['filename']}")

        await db_manager.update_collection_status(
            collection_id,
            "running",
            progress=20,
            current_query="Starting paper collection...",
        )

        # Collect papers with progress updates
        papers = []
        queries = collector.query_config["search_queries"]

        for i, query_info in enumerate(queries):
            try:
                # Update current query
                await db_manager.update_collection_status(
                    collection_id,
                    "running",
                    progress=20 + int((i / len(queries)) * 70),
                    current_query=f"Query {i+1}/{len(queries)}: {query_info['query'][:50]}...",
                )

                # Execute query
                query_papers = await collector.search_by_query(
                    query_info, request.max_papers
                )
                filtered_papers = collector.filter_papers_by_relevance(query_papers)

                # Add unique papers
                for paper in filtered_papers:
                    if paper.id not in collector.paper_ids:
                        papers.append(paper)
                        collector.paper_ids.add(paper.id)

                # Update papers found
                await db_manager.update_collection_status(
                    collection_id, "running", papers_found=len(papers)
                )

                # Rate limiting
                await asyncio.sleep(3.5)

            except Exception as e:
                print(f"Query failed: {e}")
                continue

        collector.collected_papers = papers

        # Store papers
        await db_manager.update_collection_status(
            collection_id,
            "running",
            progress=95,
            current_query="Storing papers in database...",
        )

        await collector.store_papers()

        # Mark as completed
        await db_manager.update_collection_status(
            collection_id,
            "completed",
            progress=100,
            papers_found=len(papers),
            current_query=f"Completed! Collected {len(papers)} papers.",
        )

    except Exception as e:
        # Mark as failed
        await db_manager.update_collection_status(
            collection_id, "failed", current_query=f"Error: {str(e)}"
        )


# Backup Management Endpoints


@app.post("/api/backup/create")
async def create_backup(backup_type: str = "manual"):
    """Create a manual database backup"""
    try:
        backup_service = BackupService()
        result = await backup_service.create_backup(backup_type)

        if result["success"]:
            return create_response(
                {"message": "Backup created successfully", "backup": result["metadata"]}
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup/list")
async def list_backups():
    """List all available backups"""
    try:
        backup_service = BackupService()
        backups = await backup_service.list_backups()
        return create_response(backups)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backup/restore/{backup_filename}")
async def restore_backup(backup_filename: str):
    """Restore database from backup"""
    try:
        backup_service = BackupService()
        result = await backup_service.restore_backup(backup_filename)

        if result["success"]:
            return create_response(
                {"message": "Database restored successfully", "restore_info": result}
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/backup/{backup_filename}")
async def delete_backup(backup_filename: str):
    """Delete a backup file"""
    try:
        backup_service = BackupService()
        result = await backup_service.delete_backup(backup_filename)

        if result["success"]:
            return create_response(
                {
                    "message": "Backup deleted successfully",
                    "deleted_file": result["deleted_file"],
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup/summary")
async def get_backup_summary():
    """Get backup storage summary and statistics"""
    try:
        backup_service = BackupService()
        summary = await backup_service.get_backup_summary()
        return create_response(summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backup/validate")
async def validate_backup_environment():
    """Validate backup environment and permissions"""
    try:
        backup_service = BackupService()

        # Basic validation checks
        issues = []
        warnings = []

        # Check if backup directory exists and is writable
        backup_dir = backup_service.backup_dir
        if not backup_dir.exists():
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                warnings.append("Backup directory was created")
            except Exception as e:
                issues.append(f"Cannot create backup directory: {e}")

        # Check write permissions
        test_file = backup_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"No write permission in backup directory: {e}")

        # Check database file exists
        db_manager = DatabaseManager()
        if not db_manager.db_path.exists():
            issues.append("Source database file not found")

        validation_result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "backup_directory": str(backup_dir),
            "source_database": str(db_manager.db_path),
        }

        return create_response(validation_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Graph Database Endpoints
# =============================================================================

@app.post("/api/graph/initialize")
async def initialize_graph_database():
    """Initialize graph database schema and sync from existing references."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import initialize_graph_database, sync_references_to_graph
        
        # Initialize graph schema
        graph_db = await initialize_graph_database(db_manager)
        
        # Sync existing references to graph
        sync_result = await sync_references_to_graph(db_manager)
        
        return create_response({
            "status": "completed",
            "message": "Graph database initialized and synced",
            "sync_result": sync_result
        })
        
    except Exception as e:
        logger.error(f"Error initializing graph database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/stats")
async def get_graph_statistics():
    """Get overall graph database statistics."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import GraphDatabaseManager
        
        graph_db = GraphDatabaseManager(db_manager)
        stats = await graph_db.get_graph_statistics()
        
        return create_response(stats)
        
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/subgraph/{paper_id}")
async def get_citation_subgraph(paper_id: str, depth: int = 2):
    """Get citation subgraph around a specific paper."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import GraphDatabaseManager
        
        # Handle both ROWID and ArXiv ID
        if paper_id.isdigit():
            paper = await db_manager.get_paper_by_rowid(int(paper_id))
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            arxiv_id = paper.get("id")
        else:
            arxiv_id = paper_id
        
        graph_db = GraphDatabaseManager(db_manager)
        subgraph = await graph_db.get_citation_subgraph(arxiv_id, depth)
        
        return create_response({
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "subgraph": subgraph
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting citation subgraph for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/path/{source_id}/{target_id}")
async def find_citation_path(source_id: str, target_id: str):
    """Find shortest citation path between two papers."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import GraphDatabaseManager
        
        graph_db = GraphDatabaseManager(db_manager)
        path = await graph_db.find_shortest_citation_path(source_id, target_id)
        
        return create_response({
            "source_id": source_id,
            "target_id": target_id,
            "path": path,
            "path_length": len(path) if path else 0,
            "found": path is not None
        })
        
    except Exception as e:
        logger.error(f"Error finding citation path from {source_id} to {target_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/connected-papers")
async def get_highly_connected_papers(limit: int = 20):
    """Get papers with highest citation connections."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import GraphDatabaseManager
        
        graph_db = GraphDatabaseManager(db_manager)
        papers = await graph_db.get_highly_connected_papers(limit)
        
        return create_response({
            "highly_connected_papers": papers,
            "total_papers": len(papers)
        })
        
    except Exception as e:
        logger.error(f"Error getting highly connected papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/metrics/{paper_id}")
async def get_node_metrics(paper_id: str):
    """Get network analysis metrics for a specific paper."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import GraphDatabaseManager
        
        # Handle both ROWID and ArXiv ID
        if paper_id.isdigit():
            paper = await db_manager.get_paper_by_rowid(int(paper_id))
            if not paper:
                raise HTTPException(status_code=404, detail="Paper not found")
            arxiv_id = paper.get("id")
        else:
            arxiv_id = paper_id
        
        graph_db = GraphDatabaseManager(db_manager)
        metrics = await graph_db.compute_node_metrics(arxiv_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail="Metrics not available for this paper")
        
        return create_response({
            "paper_id": paper_id,
            "arxiv_id": arxiv_id,
            "metrics": {
                "degree_centrality": metrics.degree_centrality,
                "betweenness_centrality": metrics.betweenness_centrality,
                "closeness_centrality": metrics.closeness_centrality,
                "pagerank": metrics.pagerank,
                "clustering_coefficient": metrics.clustering_coefficient,
                "computed_at": metrics.computed_at.isoformat()
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node metrics for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/sync")
async def sync_graph_data():
    """Sync graph data from existing paper references."""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        from backend.src.graph_database import sync_references_to_graph
        
        sync_result = await sync_references_to_graph(db_manager)
        
        return create_response({
            "status": "completed",
            "message": "Graph data synchronized",
            "sync_result": sync_result
        })
        
    except Exception as e:
        logger.error(f"Error syncing graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
