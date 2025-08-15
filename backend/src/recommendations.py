"""Recommendation engine for personalized arXiv paper recommendations."""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from embeddings import EmbeddingManager
from database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class RecommendationScore:
    """Container for recommendation scoring components."""

    similarity: float
    novelty: float
    diversity: float
    quality: float
    final_score: float


class RecommendationEngine:
    """
    Personalized recommendation engine using semantic similarity.

    Features:
    - Semantic similarity matching using embeddings
    - User preference learning from ratings
    - Diversity and novelty optimization
    - Quality filtering and ranking
    - Configurable scoring weights
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.db_manager = db_manager or DatabaseManager()

        # Scoring weights (can be configured via user preferences)
        self.default_weights = {
            "similarity": 0.5,  # Semantic similarity to user preferences
            "novelty": 0.2,  # How different from previously seen papers
            "diversity": 0.2,  # Diversity within recommendation set
            "quality": 0.1,  # Paper quality indicators
        }

    async def generate_recommendations(
        self,
        paper_embeddings: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations for new papers.

        Args:
            paper_embeddings: List of paper embeddings with metadata
            user_preferences: User preference dictionary
            top_k: Number of recommendations to return

        Returns:
            List of recommended papers with scores
        """
        if not paper_embeddings:
            logger.warning("No paper embeddings provided for recommendations")
            return []

        logger.info(f"Generating recommendations for {len(paper_embeddings)} papers")

        # Get user preference embedding
        user_embedding = await self._get_user_preference_embedding(user_preferences)

        # If no ratings-based embedding, derive a neutral preference from available paper embeddings
        scoring_preferences = dict(user_preferences)
        if not user_embedding:
            try:
                valid_embeddings = [
                    p["embedding"]
                    for p in paper_embeddings
                    if isinstance(p, dict) and p.get("embedding") is not None
                ]
                if valid_embeddings:
                    embeddings_array = np.array(valid_embeddings)
                    user_embedding = np.mean(embeddings_array, axis=0).tolist()
                    # Loosen similarity threshold for neutral embedding to avoid filtering out all papers
                    scoring_preferences["embedding_similarity_threshold"] = 0.0
                    logger.info(
                        f"Derived neutral user embedding from {len(valid_embeddings)} paper embeddings"
                    )
                else:
                    logger.warning(
                        "No embeddings available to derive neutral user embedding; using fallback"
                    )
                    return await self._fallback_recommendations(paper_embeddings, top_k)
            except Exception as e:
                logger.warning(f"Failed to derive neutral user embedding: {e}")
                return await self._fallback_recommendations(paper_embeddings, top_k)

        # Score all papers
        scored_papers = []
        for paper_data in paper_embeddings:
            score = await self._score_paper(
                paper_data, user_embedding, scoring_preferences
            )
            if score and score.final_score > 0:
                paper_result = {
                    "paper_id": paper_data["paper_id"],
                    "score": score.final_score,
                    "similarity": score.similarity,
                    "novelty": score.novelty,
                    "diversity": score.diversity,
                    "quality": score.quality,
                }
                scored_papers.append(paper_result)

        if not scored_papers:
            logger.warning("No papers scored above threshold")
            return []

        # Apply diversity optimization
        recommendations = self._optimize_diversity(
            scored_papers, paper_embeddings, top_k
        )

        # Store recommendations for tracking
        await self.db_manager.store_recommendations(recommendations)

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    async def _get_user_preference_embedding(
        self, user_preferences: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Get or compute user preference embedding."""
        # Try to get cached user preference embedding
        # For now, compute it from recent highly-rated papers

        user_ratings = await self.db_manager.get_user_ratings(min_rating=4)

        if not user_ratings:
            logger.info("No user ratings available for preference embedding")
            return None

        # Use embedding manager to compute preference embedding
        return await self.embedding_manager.precompute_user_preference_embedding(
            user_ratings
        )

    async def _score_paper(
        self,
        paper_data: Dict[str, Any],
        user_embedding: List[float],
        user_preferences: Dict[str, Any],
    ) -> Optional[RecommendationScore]:
        """Score a single paper against user preferences."""

        if "embedding" not in paper_data:
            logger.warning(f"No embedding found for paper {paper_data.get('paper_id')}")
            return None

        paper_embedding = paper_data["embedding"]

        # 1. Similarity score (cosine similarity with user preferences)
        similarity = self.embedding_manager.cosine_similarity(
            user_embedding, paper_embedding
        )

        # Apply similarity threshold
        min_similarity = user_preferences.get("embedding_similarity_threshold", 0.3)
        if similarity < min_similarity:
            return None

        # 2. Novelty score (how different from recently recommended papers)
        novelty = await self._calculate_novelty_score(paper_data, user_preferences)

        # 3. Quality score (based on various quality indicators)
        quality = self._calculate_quality_score(paper_data)

        # 4. Diversity score (calculated later during optimization)
        diversity = 1.0  # Placeholder, will be updated during diversity optimization

        # Combine scores using weights
        weights = self._get_scoring_weights(user_preferences)
        final_score = (
            weights["similarity"] * similarity
            + weights["novelty"] * novelty
            + weights["quality"] * quality
            + weights["diversity"] * diversity
        )

        return RecommendationScore(
            similarity=similarity,
            novelty=novelty,
            diversity=diversity,
            quality=quality,
            final_score=final_score,
        )

    def _get_scoring_weights(
        self, user_preferences: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get scoring weights from user preferences or defaults."""
        return {
            "similarity": user_preferences.get(
                "similarity_weight", self.default_weights["similarity"]
            ),
            "novelty": user_preferences.get(
                "novelty_weight", self.default_weights["novelty"]
            ),
            "diversity": user_preferences.get(
                "diversity_weight", self.default_weights["diversity"]
            ),
            "quality": user_preferences.get(
                "quality_weight", self.default_weights["quality"]
            ),
        }

    async def _calculate_novelty_score(
        self, paper_data: Dict[str, Any], user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate novelty score for a paper."""
        # For now, use a simple time-based novelty score
        # In future versions, this could compare against previously recommended papers

        try:
            # Get paper publication date if available
            published_date = paper_data.get("published_date")
            if not published_date:
                return 0.5  # Neutral novelty score

            # More recent papers get higher novelty scores
            if isinstance(published_date, str):
                published_date = datetime.fromisoformat(
                    published_date.replace("Z", "+00:00")
                )

            days_old = (datetime.now() - published_date.replace(tzinfo=None)).days

            # Exponential decay: newer papers get higher scores
            novelty = max(0.1, np.exp(-days_old / 30.0))  # 30-day half-life

            return min(1.0, novelty)

        except Exception as e:
            logger.warning(f"Error calculating novelty score: {e}")
            return 0.5

    def _calculate_quality_score(self, paper_data: Dict[str, Any]) -> float:
        """Calculate quality score for a paper based on various indicators."""
        score = 0.5  # Base score

        try:
            # Author count (papers with multiple authors might be higher quality)
            authors = paper_data.get("authors", [])
            if isinstance(authors, list) and len(authors) > 1:
                score += 0.1

            # Abstract length (papers with substantial abstracts might be higher quality)
            abstract = paper_data.get("abstract", "")
            if len(abstract) > 500:  # Substantial abstract
                score += 0.1

            # Category preferences (papers in preferred categories get bonus)
            category = paper_data.get("category", "")
            # This would be enhanced with user category preferences

            # Title indicators (avoid overly promotional language)
            title = paper_data.get("title", "").lower()
            if any(
                word in title for word in ["breakthrough", "novel", "state-of-the-art"]
            ):
                score += 0.05

            return min(1.0, score)

        except Exception as e:
            logger.warning(f"Error calculating quality score: {e}")
            return 0.5

    def _optimize_diversity(
        self,
        scored_papers: List[Dict[str, Any]],
        paper_embeddings: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Optimize recommendation diversity using MMR (Maximal Marginal Relevance)."""

        if len(scored_papers) <= top_k:
            return sorted(scored_papers, key=lambda x: x["score"], reverse=True)

        # Create embedding lookup
        embedding_lookup = {
            paper["paper_id"]: paper["embedding"]
            for paper in paper_embeddings
            if "paper_id" in paper and "embedding" in paper
        }

        # MMR algorithm
        selected = []
        candidates = scored_papers.copy()

        # Start with highest scoring paper
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected.append(candidates.pop(0))

        # Diversity weight
        lambda_param = 0.7  # Balance between relevance (1.0) and diversity (0.0)

        while len(selected) < top_k and candidates:
            best_mmr_score = -1
            best_candidate = None
            best_idx = -1

            for i, candidate in enumerate(candidates):
                candidate_id = candidate["paper_id"]
                if candidate_id not in embedding_lookup:
                    continue

                candidate_embedding = embedding_lookup[candidate_id]

                # Calculate relevance (already computed score)
                relevance = candidate["score"]

                # Calculate maximum similarity to already selected papers
                max_similarity = 0
                for selected_paper in selected:
                    selected_id = selected_paper["paper_id"]
                    if selected_id in embedding_lookup:
                        selected_embedding = embedding_lookup[selected_id]
                        similarity = self.embedding_manager.cosine_similarity(
                            candidate_embedding, selected_embedding
                        )
                        max_similarity = max(max_similarity, similarity)

                # MMR score: balance relevance and diversity
                mmr_score = (
                    lambda_param * relevance - (1 - lambda_param) * max_similarity
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = i

            if best_candidate:
                selected.append(candidates.pop(best_idx))
            else:
                # No more candidates with embeddings
                break

        # Update diversity scores
        for paper in selected:
            paper["diversity"] = 1.0 - (
                len(selected) - selected.index(paper) - 1
            ) / len(selected)

        return selected

    async def _fallback_recommendations(
        self, paper_embeddings: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback recommendations when no user preferences are available."""

        logger.info("Using fallback recommendation strategy")

        # Simple strategy: return random sample of recent papers
        shuffled_papers = paper_embeddings.copy()
        random.shuffle(shuffled_papers)

        recommendations = []
        for i, paper_data in enumerate(shuffled_papers[:top_k]):
            recommendations.append(
                {
                    "paper_id": paper_data["paper_id"],
                    "score": 0.5,  # Neutral score
                    "similarity": 0.0,
                    "novelty": 1.0,
                    "diversity": 1.0,
                    "quality": 0.5,
                    "fallback": True,
                }
            )

        return recommendations

    async def update_recommendations_feedback(self, feedback: List[Dict[str, Any]]):
        """Update recommendation algorithm based on user feedback."""

        # Store feedback ratings
        for item in feedback:
            if "paper_id" in item and "rating" in item:
                await self.db_manager.store_user_rating(
                    paper_id=item["paper_id"],
                    rating=item["rating"],
                    notes=item.get("notes"),
                )

        logger.info(f"Updated recommendations with {len(feedback)} feedback items")

    def explain_recommendation(self, paper_recommendation: Dict[str, Any]) -> str:
        """Generate explanation for why a paper was recommended."""

        explanation_parts = []

        similarity = paper_recommendation.get("similarity", 0)
        novelty = paper_recommendation.get("novelty", 0)
        quality = paper_recommendation.get("quality", 0)

        if similarity > 0.7:
            explanation_parts.append("highly similar to your interests")
        elif similarity > 0.5:
            explanation_parts.append("moderately similar to your interests")

        if novelty > 0.7:
            explanation_parts.append("recently published")

        if quality > 0.7:
            explanation_parts.append("high quality indicators")

        if paper_recommendation.get("fallback"):
            return "Recommended as part of diverse selection (no user preferences available)"

        if explanation_parts:
            return "Recommended because it is " + " and ".join(explanation_parts)
        else:
            return "Recommended based on overall scoring algorithm"

    async def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get recommendation engine statistics."""

        # Get recent recommendations
        recent_date = datetime.now() - timedelta(days=7)

        # This would query the database for recommendation statistics
        # For now, return basic stats

        return {
            "cache_stats": self.embedding_manager.get_cache_stats(),
            "algorithm_version": "v1.0",
            "scoring_weights": self.default_weights,
            "last_updated": datetime.now().isoformat(),
        }

    async def score_all_cached_papers(
        self, user_embedding: List[float], user_preferences: Dict[str, Any]
    ) -> int:
        """Compute and persist scores for all papers with cached embeddings.

        Returns the number of papers scored and stored.
        """
        try:
            # 1) Determine user embedding (ratings-based or derived from cached embeddings)
            user_embedding = await self._get_user_preference_embedding(user_preferences)

            # Fetch all paper embeddings from HDF5 cache via embedding manager
            # Build (paper_id, embedding) list by hashing each paper's abstract
            all_papers = await self.db_manager.get_all_papers()
            all_embeddings: List[Dict[str, Any]] = []
            for p in all_papers:
                try:
                    emb = await self.embedding_manager.get_embedding(
                        text=p.get("abstract", ""), paper_id=p.get("id")
                    )
                    if emb is not None:
                        all_embeddings.append(
                            {"paper_id": p.get("id"), "embedding": emb}
                        )
                except Exception:
                    continue
            if not all_embeddings:
                logger.info("No cached embeddings found; nothing to score.")
                return 0

            # If no ratings-based embedding, derive a neutral mean from cached embeddings
            scoring_preferences = dict(user_preferences)
            if not user_embedding:
                import numpy as np

                valid_embs = [
                    e["embedding"]
                    for e in all_embeddings
                    if e.get("embedding") is not None
                ]
                if not valid_embs:
                    logger.info("No valid embeddings to derive neutral user embedding.")
                    return 0
                user_embedding = (np.mean(valid_embs, axis=0)).tolist()
                scoring_preferences["embedding_similarity_threshold"] = 0.0

            # 2) Use already fetched paper metadata
            paper_meta = {p["id"]: p for p in all_papers}

            # 3) Score each paper with available embedding
            scored: List[Dict[str, Any]] = []
            for emb in all_embeddings:
                pid = emb.get("paper_id")
                if not pid:
                    continue
                meta = paper_meta.get(pid, {})
                paper_data = {
                    "paper_id": pid,
                    "embedding": emb.get("embedding"),
                    "title": meta.get("title"),
                    "abstract": meta.get("abstract"),
                    "authors": meta.get("authors"),
                    "category": meta.get("category"),
                    "published_date": meta.get("published_date"),
                }
                s = await self._score_paper(
                    paper_data, user_embedding, scoring_preferences
                )
                if s:
                    scored.append({"paper_id": pid, "score": s.final_score})

            if not scored:
                logger.info("No papers produced valid scores.")
                return 0

            # 4) Persist scores (updates current_score via store_recommendations)
            await self.db_manager.store_recommendations(
                scored, algorithm_version="bulk_v1"
            )
            logger.info(f"Bulk scored {len(scored)} papers from cached embeddings.")
            return len(scored)

        except Exception as e:
            logger.error(f"Failed bulk scoring cached papers: {e}")
            return 0
