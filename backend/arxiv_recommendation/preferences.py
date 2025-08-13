#!/usr/bin/env python3
"""
Preference Management for ArXiv Recommendation System.

This module handles:
- Recent vs all-time user preference tracking
- Exponential Moving Average (EMA) weighting for preferences
- Preference embedding computation and caching
- Adaptive preference windows
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .config import config

logger = logging.getLogger(__name__)


class PreferenceMode(Enum):
    """Preference computation modes."""
    ALL_TIME = "all_time"
    RECENT = "recent"
    ADAPTIVE = "adaptive"  # Combines recent and all-time with adaptive weighting


@dataclass
class PreferenceWindow:
    """Defines a preference time window."""
    name: str
    days: int
    weight: float
    min_ratings: int


class PreferenceManager:
    """
    Manages user preference embeddings with support for multiple time windows
    and EMA-style weighting to emphasize recent preferences.
    """

    def __init__(self):
        """Initialize the preference manager."""
        self.db_manager = DatabaseManager()
        self.embedding_manager = EmbeddingManager()
        
        # Preference computation windows
        self.windows = [
            PreferenceWindow("recent", 7, 0.7, 2),      # Last week - high weight
            PreferenceWindow("medium", 30, 0.2, 5),     # Last month - medium weight  
            PreferenceWindow("long_term", 365, 0.1, 10) # Last year - low weight
        ]
        
        # Configuration
        self.min_total_ratings = config.min_ratings_for_scoring if hasattr(config, 'min_ratings_for_scoring') else 5
        self.cache_ttl_hours = 6  # Cache preference embeddings for 6 hours
        
        # In-memory cache for preference embeddings
        self._preference_cache = {}
        self._cache_timestamps = {}

    async def get_preference_embedding(self, mode: PreferenceMode = PreferenceMode.ADAPTIVE) -> Optional[np.ndarray]:
        """
        Get user preference embedding for the specified mode.
        
        Args:
            mode: Preference computation mode (all_time, recent, or adaptive)
            
        Returns:
            Preference embedding vector or None if insufficient data
        """
        try:
            cache_key = f"preference_{mode.value}"
            
            # Check cache first
            if self._is_cached(cache_key):
                logger.debug(f"Using cached preference embedding for mode: {mode.value}")
                return self._preference_cache[cache_key]
            
            # Compute preference embedding based on mode
            if mode == PreferenceMode.ALL_TIME:
                embedding = await self._compute_all_time_preference()
            elif mode == PreferenceMode.RECENT:
                embedding = await self._compute_recent_preference()
            else:  # ADAPTIVE
                embedding = await self._compute_adaptive_preference()
            
            # Cache the result
            if embedding is not None:
                self._preference_cache[cache_key] = embedding
                self._cache_timestamps[cache_key] = datetime.utcnow()
                
                # Store in database for persistence
                await self._store_preference_embedding(mode.value, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get preference embedding for mode {mode.value}: {e}")
            return None

    async def _compute_all_time_preference(self) -> Optional[np.ndarray]:
        """Compute preference embedding using all rated papers with equal weighting."""
        try:
            # Get all rated papers
            rated_papers = await self.db_manager.get_papers_with_ratings()
            rated_papers = [p for p in rated_papers if p.get('rating', 0) > 0]
            
            if len(rated_papers) < self.min_total_ratings:
                logger.warning(f"Insufficient ratings for all-time preference: {len(rated_papers)}")
                return None
            
            return await self._compute_weighted_preference(rated_papers, use_time_weights=False)
            
        except Exception as e:
            logger.error(f"Failed to compute all-time preference: {e}")
            return None

    async def _compute_recent_preference(self, days: int = 30) -> Optional[np.ndarray]:
        """Compute preference embedding using only recent ratings."""
        try:
            # Get recent rated papers
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            rated_papers = await self.db_manager.get_papers_with_ratings()
            recent_papers = []
            
            for paper in rated_papers:
                if paper.get('rating', 0) > 0:
                    # Check if rating is recent (we need to add rating timestamp to database)
                    # For now, use paper creation date as proxy
                    created_at = paper.get('created_at')
                    if created_at:
                        if isinstance(created_at, str):
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if created_at >= cutoff_date:
                            recent_papers.append(paper)
            
            if len(recent_papers) < 3:  # Lower threshold for recent
                logger.warning(f"Insufficient recent ratings: {len(recent_papers)}")
                return None
            
            return await self._compute_weighted_preference(recent_papers, use_time_weights=True)
            
        except Exception as e:
            logger.error(f"Failed to compute recent preference: {e}")
            return None

    async def _compute_adaptive_preference(self) -> Optional[np.ndarray]:
        """
        Compute adaptive preference embedding using EMA-style weighting
        that combines multiple time windows.
        """
        try:
            # Get all rated papers with timestamps
            rated_papers = await self.db_manager.get_papers_with_ratings()
            rated_papers = [p for p in rated_papers if p.get('rating', 0) > 0]
            
            if len(rated_papers) < self.min_total_ratings:
                logger.warning(f"Insufficient ratings for adaptive preference: {len(rated_papers)}")
                return None
            
            # Organize papers by time windows
            now = datetime.utcnow()
            window_papers = {window.name: [] for window in self.windows}
            
            for paper in rated_papers:
                created_at = paper.get('created_at')
                if created_at:
                    if isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    
                    age_days = (now - created_at).days
                    
                    # Assign to appropriate windows
                    for window in self.windows:
                        if age_days <= window.days:
                            window_papers[window.name].append(paper)
            
            # Compute embeddings for each window
            window_embeddings = []
            window_weights = []
            
            for window in self.windows:
                papers = window_papers[window.name]
                if len(papers) >= window.min_ratings:
                    try:
                        embedding = await self._compute_weighted_preference(papers, use_time_weights=True)
                        if embedding is not None:
                            window_embeddings.append(embedding)
                            window_weights.append(window.weight)
                            logger.debug(f"Window {window.name}: {len(papers)} papers, weight {window.weight}")
                    except Exception as e:
                        logger.warning(f"Failed to compute embedding for window {window.name}: {e}")
            
            if not window_embeddings:
                logger.warning("No valid window embeddings computed")
                return None
            
            # Combine window embeddings with weights
            if len(window_embeddings) == 1:
                return window_embeddings[0]
            
            # Normalize weights
            total_weight = sum(window_weights)
            normalized_weights = [w / total_weight for w in window_weights]
            
            # Compute weighted average
            embeddings_array = np.array(window_embeddings)
            adaptive_embedding = np.average(embeddings_array, axis=0, weights=normalized_weights)
            
            logger.info(f"Computed adaptive preference from {len(window_embeddings)} time windows")
            return adaptive_embedding
            
        except Exception as e:
            logger.error(f"Failed to compute adaptive preference: {e}")
            return None

    async def _compute_weighted_preference(self, papers: List[Dict[str, Any]], use_time_weights: bool = True) -> Optional[np.ndarray]:
        """
        Compute weighted preference embedding from a list of rated papers.
        
        Args:
            papers: List of rated papers
            use_time_weights: Whether to apply time-based decay weights
        """
        try:
            paper_embeddings = []
            rating_weights = []
            time_weights = []
            
            now = datetime.utcnow()
            
            for paper in papers:
                try:
                    # Get paper embedding
                    text = f"{paper['title']} {paper['abstract']}"
                    embedding = await self.embedding_manager.get_embedding(text)
                    
                    if embedding is not None:
                        paper_embeddings.append(embedding)
                        
                        # Rating weight (exponential weighting favors higher ratings)
                        rating = paper.get('rating', 0)
                        rating_weight = np.exp((rating - 1) / 4)  # Convert 1-5 to exponential weight
                        rating_weights.append(rating_weight)
                        
                        # Time weight (recent papers get higher weights if enabled)
                        if use_time_weights:
                            created_at = paper.get('created_at')
                            if created_at and isinstance(created_at, str):
                                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                age_days = (now - created_at).days
                                # Exponential decay: weight = exp(-age_days / decay_constant)
                                time_weight = np.exp(-age_days / 30)  # 30-day decay constant
                            else:
                                time_weight = 1.0
                        else:
                            time_weight = 1.0
                        
                        time_weights.append(time_weight)
                
                except Exception as e:
                    logger.warning(f"Failed to process paper {paper.get('id', 'unknown')}: {e}")
                    continue
            
            if not paper_embeddings:
                logger.error("No valid paper embeddings found")
                return None
            
            # Combine rating and time weights
            embeddings_array = np.array(paper_embeddings)
            rating_weights = np.array(rating_weights)
            time_weights = np.array(time_weights)
            
            # Combined weights (multiplicative)
            combined_weights = rating_weights * time_weights
            combined_weights = combined_weights / combined_weights.sum()  # Normalize
            
            # Compute weighted average
            preference_embedding = np.average(embeddings_array, axis=0, weights=combined_weights)
            
            logger.debug(f"Computed preference from {len(paper_embeddings)} papers")
            return preference_embedding
            
        except Exception as e:
            logger.error(f"Failed to compute weighted preference: {e}")
            return None

    async def _store_preference_embedding(self, mode: str, embedding: np.ndarray):
        """Store preference embedding in database for persistence."""
        try:
            # Convert embedding to JSON for storage
            embedding_data = {
                "mode": mode,
                "embedding": embedding.tolist(),
                "created_at": datetime.utcnow().isoformat(),
                "dimensions": len(embedding)
            }
            
            await self.db_manager.update_user_preference(
                key=f"preference_embedding_{mode}",
                value=embedding_data,
                value_type="json"
            )
            
            logger.debug(f"Stored preference embedding for mode: {mode}")
            
        except Exception as e:
            logger.error(f"Failed to store preference embedding for mode {mode}: {e}")

    async def load_stored_preference_embedding(self, mode: str) -> Optional[np.ndarray]:
        """Load stored preference embedding from database."""
        try:
            preferences = await self.db_manager.get_user_preferences()
            embedding_key = f"preference_embedding_{mode}"
            
            if embedding_key in preferences:
                embedding_data = preferences[embedding_key]
                
                # Check if embedding is recent enough
                created_at = datetime.fromisoformat(embedding_data["created_at"])
                age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
                
                if age_hours <= self.cache_ttl_hours:
                    embedding = np.array(embedding_data["embedding"])
                    logger.debug(f"Loaded stored preference embedding for mode: {mode}")
                    return embedding
                else:
                    logger.debug(f"Stored preference embedding for mode {mode} is expired ({age_hours:.1f}h old)")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load stored preference embedding for mode {mode}: {e}")
            return None

    async def update_preference_embeddings(self) -> Dict[str, Any]:
        """
        Update all preference embeddings.
        
        This is typically called by the scheduler to refresh preference embeddings
        after new ratings are added.
        """
        try:
            logger.info("Updating preference embeddings")
            start_time = datetime.utcnow()
            
            results = {
                "embeddings_updated": 0,
                "modes_processed": [],
                "errors": []
            }
            
            # Clear cache to force recomputation
            self._preference_cache.clear()
            self._cache_timestamps.clear()
            
            # Update embeddings for all modes
            for mode in PreferenceMode:
                try:
                    embedding = await self.get_preference_embedding(mode)
                    if embedding is not None:
                        results["embeddings_updated"] += 1
                        results["modes_processed"].append(mode.value)
                        logger.info(f"Updated preference embedding for mode: {mode.value}")
                    else:
                        logger.warning(f"Could not compute preference embedding for mode: {mode.value}")
                        
                except Exception as e:
                    error_msg = f"Failed to update preference embedding for mode {mode.value}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Update processing time
            results["processing_time_seconds"] = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Preference embedding update completed: {results['embeddings_updated']} embeddings updated")
            return results
            
        except Exception as e:
            logger.error(f"Preference embedding update failed: {e}")
            return {
                "embeddings_updated": 0,
                "modes_processed": [],
                "errors": [str(e)]
            }

    def _is_cached(self, cache_key: str) -> bool:
        """Check if a preference embedding is cached and still valid."""
        if cache_key not in self._preference_cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key)
        if not timestamp:
            return False
        
        age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
        return age_hours <= self.cache_ttl_hours

    async def get_preference_mode(self) -> PreferenceMode:
        """Get the current preference mode from user settings."""
        try:
            preferences = await self.db_manager.get_user_preferences()
            mode_str = preferences.get("preference_mode", "adaptive")
            
            # Convert string to enum
            for mode in PreferenceMode:
                if mode.value == mode_str:
                    return mode
            
            # Default to adaptive if invalid mode
            return PreferenceMode.ADAPTIVE
            
        except Exception as e:
            logger.error(f"Failed to get preference mode: {e}")
            return PreferenceMode.ADAPTIVE

    async def set_preference_mode(self, mode: PreferenceMode):
        """Set the current preference mode in user settings."""
        try:
            await self.db_manager.update_user_preference(
                key="preference_mode",
                value=mode.value,
                value_type="string"
            )
            
            # Clear cache to force recomputation with new mode
            self._preference_cache.clear()
            self._cache_timestamps.clear()
            
            logger.info(f"Preference mode set to: {mode.value}")
            
        except Exception as e:
            logger.error(f"Failed to set preference mode to {mode.value}: {e}")
            raise

    def get_preference_stats(self) -> Dict[str, Any]:
        """Get preference management statistics."""
        return {
            "available_modes": [mode.value for mode in PreferenceMode],
            "cache_size": len(self._preference_cache),
            "cached_modes": list(self._preference_cache.keys()),
            "cache_ttl_hours": self.cache_ttl_hours,
            "min_total_ratings": self.min_total_ratings,
            "time_windows": [
                {
                    "name": w.name,
                    "days": w.days,
                    "weight": w.weight,
                    "min_ratings": w.min_ratings
                } for w in self.windows
            ]
        }