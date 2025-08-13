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

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.arxiv_categories is None:
            categories_str = os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG,cs.CL")
            self.arxiv_categories = [cat.strip() for cat in categories_str.split(",")]

        # Ensure data directories exist
        for path in [self.embeddings_path, Path(self.database_path).parent]:
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
