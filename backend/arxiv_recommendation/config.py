"""Configuration management for ArXiv recommendation system."""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the ArXiv recommendation system."""

    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = "text-embedding-3-small"  # Cost-effective
    max_tokens_per_request: int = 8000

    # ArXiv Configuration
    arxiv_categories: List[str] = None
    max_daily_papers: int = int(os.getenv("MAX_DAILY_PAPERS", "100"))
    arxiv_base_url: str = "https://export.arxiv.org/api/query"

    # Database Configuration
    database_path: str = "data/papers.db"
    embeddings_path: str = "data/embeddings"
    user_ratings_path: str = "data/user_ratings.json"

    # Cost Management
    openai_budget_limit: float = float(os.getenv("OPENAI_BUDGET_LIMIT", "20.0"))
    cost_tracking_enabled: bool = True

    # Agent Configuration
    agent_timeout: int = 30
    max_retries: int = 3

    # Caching
    cache_embeddings: bool = True
    embedding_cache_ttl: int = 30 * 24 * 3600  # 30 days

    # Scheduler Configuration
    collection_time: str = os.getenv("COLLECTION_TIME", "06:00")  # Daily collection time (HH:MM)
    scoring_time: str = os.getenv("SCORING_TIME", "07:00")  # Daily scoring time (HH:MM)
    preference_update_time: str = os.getenv("PREFERENCE_TIME", "08:00")  # Preference update time
    cache_maintenance_interval: int = int(os.getenv("CACHE_MAINTENANCE_HOURS", "6"))  # Hours
    
    # Collection Strategy
    collection_topics: List[str] = None  # Will be set in __post_init__
    max_papers_per_query: int = int(os.getenv("MAX_PAPERS_PER_QUERY", "100"))
    query_cache_ttl_hours: int = int(os.getenv("QUERY_CACHE_TTL", "24"))
    rate_limit_delay: float = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))  # seconds between API calls
    
    # Scoring Configuration  
    min_ratings_for_scoring: int = int(os.getenv("MIN_RATINGS_FOR_SCORING", "5"))
    score_cache_ttl_hours: int = int(os.getenv("SCORE_CACHE_TTL", "24"))
    
    # Preference Management
    preference_cache_ttl_hours: int = int(os.getenv("PREFERENCE_CACHE_TTL", "6"))
    recent_preference_days: int = int(os.getenv("RECENT_PREFERENCE_DAYS", "30"))
    preference_decay_constant: float = float(os.getenv("PREFERENCE_DECAY_CONSTANT", "30.0"))  # days
    
    # Scheduler Database
    scheduler_db_path: str = "data/scheduler.db"

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.arxiv_categories is None:
            categories_str = os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL")
            self.arxiv_categories = [cat.strip() for cat in categories_str.split(",")]
        
        if self.collection_topics is None:
            topics_str = os.getenv("COLLECTION_TOPICS", "machine learning,deep learning,natural language processing")
            self.collection_topics = [topic.strip() for topic in topics_str.split(",")]

        # Ensure data directories exist
        for path in [self.embeddings_path, Path(self.database_path).parent, Path(self.scheduler_db_path).parent]:
            Path(path).mkdir(parents=True, exist_ok=True)

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

    @property
    def embedding_cost_per_token(self) -> float:
        """Cost per token for the selected embedding model."""
        costs = {
            "text-embedding-3-small": 0.00002 / 1000,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013 / 1000,  # $0.00013 per 1K tokens
        }
        return costs.get(self.embedding_model, 0.00002 / 1000)

    def estimate_daily_cost(self) -> float:
        """Estimate daily OpenAI API costs based on configuration."""
        avg_tokens_per_abstract = 150
        daily_tokens = self.max_daily_papers * avg_tokens_per_abstract
        return daily_tokens * self.embedding_cost_per_token


# Global configuration instance
config = Config()
