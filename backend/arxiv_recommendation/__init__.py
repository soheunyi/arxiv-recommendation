"""ArXiv Recommendation System - Personal LLM-based paper recommendations."""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Personal arXiv recommendation system with LLM embeddings"

from .config import Config
from .agents import DataAgent, RecommendationAgent, Coordinator, run_recommendation_system

__all__ = ["Config", "DataAgent", "RecommendationAgent", "Coordinator", "run_recommendation_system"]