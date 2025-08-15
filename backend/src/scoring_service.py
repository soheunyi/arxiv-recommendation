#!/usr/bin/env python3
"""
Scoring Service for ArXiv Recommendation System.

This service handles:
- Computing recommendation scores for papers
- Updating daily scores using cached embeddings
- Managing user preference embeddings
- Efficient batch processing of scores
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from database import DatabaseManager
from embeddings import EmbeddingManager
from config import config

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of a scoring operation."""
    papers_scored: int
    papers_skipped: int
    processing_time_seconds: float
    embedding_cache_hits: int
    embedding_cache_misses: int
    errors: List[str]


class ScoringService:
    """
    Service for computing and updating paper recommendation scores.
    
    Uses cached embeddings and efficient batch processing to score papers
    based on user preferences derived from ratings.
    """

    def __init__(self):
        """Initialize the scoring service."""
        self.db_manager = DatabaseManager()
        self.embedding_manager = EmbeddingManager()
        
        # Scoring parameters
        self.min_ratings_for_scoring = config.min_ratings_for_scoring if hasattr(config, 'min_ratings_for_scoring') else 3
        self.score_cache_ttl_hours = 24  # Rescore papers after 24 hours
        
        # Performance tracking
        self.stats = {
            "total_scored": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": []
        }

    async def update_daily_scores(self) -> ScoringResult:
        """
        Update recommendation scores for papers that need rescoring.
        
        This includes:
        - Newly added papers
        - Papers not scored in the last 24 hours
        - Papers where user preferences may have changed
        """
        start_time = datetime.utcnow()
        logger.info("Starting daily score update")
        
        try:
            # Initialize services
            await self.db_manager.initialize()
            
            # Get user preference embedding
            preference_embedding = await self._get_user_preference_embedding()
            if preference_embedding is None:
                logger.warning("Cannot compute scores - insufficient user ratings")
                return ScoringResult(0, 0, 0, 0, 0, ["Insufficient user ratings for scoring"])
            
            # Get papers that need scoring
            papers_to_score = await self._get_papers_needing_scores()
            logger.info(f"Found {len(papers_to_score)} papers needing scores")
            
            if not papers_to_score:
                return ScoringResult(0, 0, 0, 0, 0, [])
            
            # Process papers in batches for efficiency
            batch_size = 50
            total_scored = 0
            total_skipped = 0
            errors = []
            
            for i in range(0, len(papers_to_score), batch_size):
                batch = papers_to_score[i:i + batch_size]
                
                try:
                    batch_result = await self._score_paper_batch(batch, preference_embedding)
                    total_scored += batch_result['scored']
                    total_skipped += batch_result['skipped']
                    
                    if batch_result['errors']:
                        errors.extend(batch_result['errors'])
                    
                    logger.info(f"Processed batch {i // batch_size + 1}: "
                              f"{batch_result['scored']} scored, {batch_result['skipped']} skipped")
                    
                except Exception as e:
                    error_msg = f"Failed to process batch starting at {i}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    total_skipped += len(batch)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update global stats
            self.stats["total_scored"] += total_scored
            self.stats["cache_hits"] += self.embedding_manager.cache_hits
            self.stats["cache_misses"] += self.embedding_manager.cache_misses
            self.stats["errors"].extend(errors)
            
            logger.info(f"Daily score update completed: {total_scored} papers scored, "
                       f"{total_skipped} skipped in {processing_time:.2f}s")
            
            return ScoringResult(
                papers_scored=total_scored,
                papers_skipped=total_skipped,
                processing_time_seconds=processing_time,
                embedding_cache_hits=self.embedding_manager.cache_hits,
                embedding_cache_misses=self.embedding_manager.cache_misses,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Daily score update failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def _get_user_preference_embedding(self) -> Optional[np.ndarray]:
        """
        Compute user preference embedding from rated papers.
        
        Uses weighted average of embeddings from rated papers,
        with higher-rated papers having more influence.
        """
        try:
            # Get all rated papers
            rated_papers = await self.db_manager.get_papers_with_ratings()
            rated_papers = [p for p in rated_papers if p.get('rating', 0) > 0]
            
            if len(rated_papers) < self.min_ratings_for_scoring:
                logger.warning(f"Only {len(rated_papers)} rated papers, need {self.min_ratings_for_scoring}")
                return None
            
            # Get embeddings for rated papers
            paper_embeddings = []
            ratings = []
            
            for paper in rated_papers:
                try:
                    # Get text to embed (title + abstract)
                    text = f"{paper['title']} {paper['abstract']}"
                    
                    # Get cached embedding
                    embedding = await self.embedding_manager.get_embedding(text)
                    if embedding is not None:
                        paper_embeddings.append(embedding)
                        ratings.append(paper['rating'])
                    
                except Exception as e:
                    logger.warning(f"Failed to get embedding for paper {paper.get('id', 'unknown')}: {e}")
                    continue
            
            if not paper_embeddings:
                logger.error("No embeddings available for rated papers")
                return None
            
            # Convert to numpy arrays
            embeddings_array = np.array(paper_embeddings)
            ratings_array = np.array(ratings)
            
            # Apply rating weights (normalize ratings to 0-1, then apply exponential weighting)
            normalized_ratings = (ratings_array - 1) / 4  # Convert 1-5 to 0-1
            weights = np.exp(normalized_ratings)  # Exponential weighting favors higher ratings
            weights = weights / weights.sum()  # Normalize weights
            
            # Compute weighted average
            preference_embedding = np.average(embeddings_array, axis=0, weights=weights)
            
            logger.info(f"Computed preference embedding from {len(paper_embeddings)} rated papers")
            return preference_embedding
            
        except Exception as e:
            logger.error(f"Failed to compute user preference embedding: {e}")
            return None

    async def _get_papers_needing_scores(self) -> List[Dict[str, Any]]:
        """
        Get papers that need score updates.
        
        This includes:
        - Papers never scored (current_score is NULL)
        - Papers not scored in the last 24 hours
        - Papers added in the last 48 hours (to ensure new papers get scored)
        """
        try:
            # Get cutoff time for stale scores
            score_cutoff = datetime.utcnow() - timedelta(hours=self.score_cache_ttl_hours)
            
            # Query for papers needing scores
            papers = await self.db_manager.get_papers_needing_scores(score_cutoff)
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to get papers needing scores: {e}")
            return []

    async def _score_paper_batch(self, papers: List[Dict[str, Any]], preference_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Score a batch of papers using the preference embedding.
        
        Returns dict with 'scored', 'skipped', and 'errors' counts.
        """
        scored_count = 0
        skipped_count = 0
        errors = []
        
        # Prepare batch data
        paper_ids = []
        paper_texts = []
        paper_embeddings = []
        
        # Get embeddings for all papers in batch
        for paper in papers:
            try:
                paper_id = paper['id']
                text = f"{paper['title']} {paper['abstract']}"
                
                # Get cached embedding
                embedding = await self.embedding_manager.get_embedding(text)
                if embedding is not None:
                    paper_ids.append(paper_id)
                    paper_texts.append(text)
                    paper_embeddings.append(embedding)
                else:
                    logger.warning(f"No embedding available for paper {paper_id}")
                    skipped_count += 1
                
            except Exception as e:
                error_msg = f"Error processing paper {paper.get('id', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                skipped_count += 1
        
        if not paper_embeddings:
            return {'scored': 0, 'skipped': skipped_count, 'errors': errors}
        
        # Compute similarity scores in batch
        try:
            embeddings_array = np.array(paper_embeddings)
            
            # Compute cosine similarities
            scores = self._compute_cosine_similarities(embeddings_array, preference_embedding)
            
            # Update database with scores
            updates = []
            for i, (paper_id, score) in enumerate(zip(paper_ids, scores)):
                # Convert numpy float to Python float and clamp to reasonable range
                score_value = float(np.clip(score, -1.0, 1.0))
                
                updates.append({
                    'paper_id': paper_id,
                    'score': score_value,
                    'score_updated_at': datetime.utcnow()
                })
            
            # Batch update database
            await self.db_manager.update_paper_scores(updates)
            scored_count = len(updates)
            
            logger.debug(f"Scored {scored_count} papers in batch")
            
        except Exception as e:
            error_msg = f"Error computing scores for batch: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            skipped_count += len(paper_ids)
        
        return {
            'scored': scored_count,
            'skipped': skipped_count,
            'errors': errors
        }

    def _compute_cosine_similarities(self, embeddings: np.ndarray, preference_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between paper embeddings and preference embedding.
        
        Args:
            embeddings: Array of shape (n_papers, embedding_dim)
            preference_embedding: Array of shape (embedding_dim,)
            
        Returns:
            Array of cosine similarities of shape (n_papers,)
        """
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        preference_norm = preference_embedding / (np.linalg.norm(preference_embedding) + 1e-8)
        
        # Compute cosine similarities
        similarities = np.dot(embeddings_norm, preference_norm)
        
        return similarities

    async def score_single_paper(self, paper_id: str) -> Optional[float]:
        """
        Score a single paper and update the database.
        
        Returns the computed score or None if scoring failed.
        """
        try:
            # Get user preference embedding
            preference_embedding = await self._get_user_preference_embedding()
            if preference_embedding is None:
                logger.warning("Cannot score paper - insufficient user ratings")
                return None
            
            # Get paper data
            paper = await self.db_manager.get_paper_by_id(paper_id)
            if not paper:
                logger.error(f"Paper {paper_id} not found")
                return None
            
            # Get paper embedding
            text = f"{paper['title']} {paper['abstract']}"
            paper_embedding = await self.embedding_manager.get_embedding(text)
            
            if paper_embedding is None:
                logger.error(f"Could not get embedding for paper {paper_id}")
                return None
            
            # Compute score
            embeddings_array = np.array([paper_embedding])
            scores = self._compute_cosine_similarities(embeddings_array, preference_embedding)
            score = float(scores[0])
            
            # Update database
            await self.db_manager.update_paper_scores([{
                'paper_id': paper_id,
                'score': score,
                'score_updated_at': datetime.utcnow()
            }])
            
            logger.info(f"Scored paper {paper_id}: {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"Failed to score paper {paper_id}: {e}")
            return None

    def get_scoring_stats(self) -> Dict[str, Any]:
        """Get scoring service statistics."""
        return {
            "total_papers_scored": self.stats["total_scored"],
            "embedding_cache_hits": self.stats["cache_hits"],
            "embedding_cache_misses": self.stats["cache_misses"],
            "recent_errors": self.stats["errors"][-10:],  # Last 10 errors
            "min_ratings_required": self.min_ratings_for_scoring,
            "score_cache_ttl_hours": self.score_cache_ttl_hours
        }