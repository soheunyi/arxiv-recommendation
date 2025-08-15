#!/usr/bin/env python3
"""
Collaborative Service for ArXiv Recommendation System.

This service orchestrates collaboration between OpenAI and Gemini for optimal
query generation, cost efficiency, and quality assurance.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from config import config
from services.provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


class CollaborationStrategy(Enum):
    """Strategies for OpenAI-Gemini collaboration."""
    COST_OPTIMIZED = "cost_optimized"  # Prefer Gemini, OpenAI for validation
    QUALITY_FIRST = "quality_first"    # OpenAI primary, Gemini backup
    PARALLEL_COMPARE = "parallel_compare"  # Both providers, compare results
    ADAPTIVE = "adaptive"              # Dynamic strategy based on task


@dataclass
class CollaborationResult:
    """Result from collaborative query generation."""
    primary_result: Dict[str, Any]
    secondary_result: Optional[Dict[str, Any]] = None
    primary_provider: str = ""
    secondary_provider: Optional[str] = None
    strategy_used: str = ""
    cost_estimate: float = 0.0
    quality_score: float = 0.0
    execution_time: float = 0.0


class CollaborativeService:
    """
    Service for orchestrating collaboration between OpenAI and Gemini.
    
    Features:
    - Intelligent provider selection based on task requirements
    - Cost optimization with quality assurance
    - Parallel execution and result comparison
    - Fallback mechanisms for reliability
    """
    
    def __init__(self):
        self.provider_factory = ProviderFactory()
        self.usage_stats = {
            "openai": {"requests": 0, "total_cost": 0.0},
            "gemini": {"requests": 0, "total_cost": 0.0}
        }
    
    async def generate_queries_collaborative(
        self,
        topic: str,
        max_queries: int = 15,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        strategy: CollaborationStrategy = CollaborationStrategy.ADAPTIVE,
        quality_threshold: float = 0.8
    ) -> CollaborationResult:
        """
        Generate queries using collaborative OpenAI-Gemini approach.
        
        Args:
            topic: Research topic
            max_queries: Maximum number of queries
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            strategy: Collaboration strategy
            quality_threshold: Minimum quality score (0.0-1.0)
            
        Returns:
            CollaborationResult with query generation results
        """
        start_time = asyncio.get_event_loop().time()
        
        # Select strategy if adaptive
        if strategy == CollaborationStrategy.ADAPTIVE:
            strategy = self._select_adaptive_strategy(topic, max_queries)
        
        logger.info(f"Using collaboration strategy: {strategy.value} for topic: '{topic}'")
        
        # Execute based on strategy
        if strategy == CollaborationStrategy.COST_OPTIMIZED:
            result = await self._cost_optimized_generation(
                topic, max_queries, date_from, date_to, quality_threshold
            )
        elif strategy == CollaborationStrategy.QUALITY_FIRST:
            result = await self._quality_first_generation(
                topic, max_queries, date_from, date_to, quality_threshold
            )
        elif strategy == CollaborationStrategy.PARALLEL_COMPARE:
            result = await self._parallel_comparison_generation(
                topic, max_queries, date_from, date_to
            )
        else:
            # Fallback to cost optimized
            result = await self._cost_optimized_generation(
                topic, max_queries, date_from, date_to, quality_threshold
            )
        
        result.execution_time = asyncio.get_event_loop().time() - start_time
        result.strategy_used = strategy.value
        
        # Update usage statistics
        self._update_usage_stats(result)
        
        logger.info(
            f"Collaborative generation completed: "
            f"strategy={strategy.value}, time={result.execution_time:.2f}s, "
            f"quality={result.quality_score:.2f}, cost=${result.cost_estimate:.4f}"
        )
        
        return result
    
    async def _cost_optimized_generation(
        self,
        topic: str,
        max_queries: int,
        date_from: Optional[str],
        date_to: Optional[str],
        quality_threshold: float
    ) -> CollaborationResult:
        """Cost-optimized strategy: Gemini primary, OpenAI validation if needed."""
        
        # Primary: Use Gemini (much cheaper)
        gemini_service = self.provider_factory.create_query_service("gemini")
        primary_result = gemini_service.generate_search_queries(
            topic, max_queries, date_from, date_to
        )
        
        quality_score = self._evaluate_query_quality(primary_result)
        cost_estimate = self._estimate_cost("gemini", topic, max_queries)
        
        result = CollaborationResult(
            primary_result=primary_result,
            primary_provider="gemini",
            quality_score=quality_score,
            cost_estimate=cost_estimate
        )
        
        # If quality is below threshold, use OpenAI for validation/improvement
        if quality_score < quality_threshold:
            logger.info(f"Quality {quality_score:.2f} below threshold {quality_threshold}, using OpenAI validation")
            
            openai_service = self.provider_factory.create_query_service("openai")
            secondary_result = openai_service.generate_search_queries(
                topic, max_queries, date_from, date_to
            )
            
            # Merge results for best quality
            merged_result = self._merge_query_results(primary_result, secondary_result)
            result.primary_result = merged_result
            result.secondary_result = secondary_result
            result.secondary_provider = "openai"
            result.quality_score = self._evaluate_query_quality(merged_result)
            result.cost_estimate += self._estimate_cost("openai", topic, max_queries)
        
        return result
    
    async def _quality_first_generation(
        self,
        topic: str,
        max_queries: int,
        date_from: Optional[str],
        date_to: Optional[str],
        quality_threshold: float
    ) -> CollaborationResult:
        """Quality-first strategy: OpenAI primary, Gemini backup."""
        
        try:
            # Primary: Use OpenAI (higher quality)
            openai_service = self.provider_factory.create_query_service("openai")
            primary_result = openai_service.generate_search_queries(
                topic, max_queries, date_from, date_to
            )
            
            quality_score = self._evaluate_query_quality(primary_result)
            cost_estimate = self._estimate_cost("openai", topic, max_queries)
            
            return CollaborationResult(
                primary_result=primary_result,
                primary_provider="openai",
                quality_score=quality_score,
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            logger.warning(f"OpenAI failed, falling back to Gemini: {e}")
            
            # Fallback: Use Gemini
            gemini_service = self.provider_factory.create_query_service("gemini")
            fallback_result = gemini_service.generate_search_queries(
                topic, max_queries, date_from, date_to
            )
            
            return CollaborationResult(
                primary_result=fallback_result,
                primary_provider="gemini",
                quality_score=self._evaluate_query_quality(fallback_result),
                cost_estimate=self._estimate_cost("gemini", topic, max_queries)
            )
    
    async def _parallel_comparison_generation(
        self,
        topic: str,
        max_queries: int,
        date_from: Optional[str],
        date_to: Optional[str]
    ) -> CollaborationResult:
        """Parallel comparison: Both providers, select best result."""
        
        # Run both providers in parallel
        async def run_openai():
            openai_service = self.provider_factory.create_query_service("openai")
            return openai_service.generate_search_queries(topic, max_queries, date_from, date_to)
        
        async def run_gemini():
            gemini_service = self.provider_factory.create_query_service("gemini")
            return gemini_service.generate_search_queries(topic, max_queries, date_from, date_to)
        
        try:
            openai_result, gemini_result = await asyncio.gather(
                run_openai(), run_gemini(), return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(openai_result, Exception):
                logger.warning(f"OpenAI failed: {openai_result}")
                openai_result = None
            
            if isinstance(gemini_result, Exception):
                logger.warning(f"Gemini failed: {gemini_result}")
                gemini_result = None
            
            # Select best result
            if openai_result and gemini_result:
                openai_quality = self._evaluate_query_quality(openai_result)
                gemini_quality = self._evaluate_query_quality(gemini_result)
                
                if openai_quality >= gemini_quality:
                    primary_result, primary_provider = openai_result, "openai"
                    secondary_result, secondary_provider = gemini_result, "gemini"
                    quality_score = openai_quality
                else:
                    primary_result, primary_provider = gemini_result, "gemini"
                    secondary_result, secondary_provider = openai_result, "openai"
                    quality_score = gemini_quality
                
                # Merge for even better quality
                merged_result = self._merge_query_results(primary_result, secondary_result)
                
                return CollaborationResult(
                    primary_result=merged_result,
                    secondary_result=secondary_result,
                    primary_provider=primary_provider,
                    secondary_provider=secondary_provider,
                    quality_score=self._evaluate_query_quality(merged_result),
                    cost_estimate=(
                        self._estimate_cost("openai", topic, max_queries) +
                        self._estimate_cost("gemini", topic, max_queries)
                    )
                )
            
            elif openai_result:
                return CollaborationResult(
                    primary_result=openai_result,
                    primary_provider="openai",
                    quality_score=self._evaluate_query_quality(openai_result),
                    cost_estimate=self._estimate_cost("openai", topic, max_queries)
                )
            
            elif gemini_result:
                return CollaborationResult(
                    primary_result=gemini_result,
                    primary_provider="gemini",
                    quality_score=self._evaluate_query_quality(gemini_result),
                    cost_estimate=self._estimate_cost("gemini", topic, max_queries)
                )
            
            else:
                raise Exception("Both providers failed")
                
        except Exception as e:
            logger.error(f"Parallel comparison failed: {e}")
            # Fallback to cost optimized
            return await self._cost_optimized_generation(topic, max_queries, date_from, date_to, 0.5)
    
    def _select_adaptive_strategy(self, topic: str, max_queries: int) -> CollaborationStrategy:
        """Select the best strategy based on topic complexity and requirements."""
        
        # Simple complexity scoring
        complexity_indicators = [
            "deep learning", "neural network", "transformer", "attention",
            "quantum", "optimization", "algorithm", "theory"
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in topic.lower())
        
        # Check current usage and budget
        total_cost = sum(stats["total_cost"] for stats in self.usage_stats.values())
        budget_used = total_cost / config.openai_budget_limit
        
        # Decision logic
        if budget_used > 0.8:  # High budget usage
            return CollaborationStrategy.COST_OPTIMIZED
        elif complexity_score >= 3 or max_queries > 20:  # High complexity
            return CollaborationStrategy.PARALLEL_COMPARE
        elif complexity_score >= 1:  # Medium complexity
            return CollaborationStrategy.QUALITY_FIRST
        else:  # Low complexity
            return CollaborationStrategy.COST_OPTIMIZED
    
    def _evaluate_query_quality(self, result: Dict[str, Any]) -> float:
        """Evaluate query quality based on various metrics."""
        if not result or "search_queries" not in result:
            return 0.0
        
        queries = result["search_queries"]
        if not queries:
            return 0.0
        
        score = 0.0
        
        # Query count score (more queries = better coverage)
        query_count_score = min(len(queries) / 15.0, 1.0) * 0.25
        score += query_count_score
        
        # Query complexity score (more sophisticated queries = better)
        complexity_score = 0.0
        for query in queries:
            query_text = query.get("query", "")
            # Award points for sophisticated query structure
            if " AND " in query_text or " OR " in query_text:
                complexity_score += 0.1
            if any(field in query_text for field in ["ti:", "abs:", "au:", "cat:"]):
                complexity_score += 0.1
            if "[" in query_text and "TO" in query_text:  # Date ranges
                complexity_score += 0.1
        
        complexity_score = min(complexity_score / len(queries), 0.3) * 0.25
        score += complexity_score
        
        # Query diversity score (different types of queries)
        query_types = set()
        for query in queries:
            query_text = query.get("query", "")
            if "ti:" in query_text:
                query_types.add("title")
            if "abs:" in query_text:
                query_types.add("abstract")
            if "au:" in query_text:
                query_types.add("author")
            if "cat:" in query_text:
                query_types.add("category")
        
        diversity_score = len(query_types) / 4.0 * 0.25
        score += diversity_score
        
        # Priority distribution score (good mix of priorities)
        priorities = [q.get("priority", "medium") for q in queries]
        priority_counts = {p: priorities.count(p) for p in ["high", "medium", "low"]}
        priority_score = 0.15 if all(count > 0 for count in priority_counts.values()) else 0.08
        score += priority_score
        
        # Metadata completeness score
        has_categories = bool(result.get("categories"))
        has_keywords = bool(result.get("filter_keywords"))
        has_related_terms = bool(result.get("related_terms"))
        metadata_score = (has_categories + has_keywords + has_related_terms) / 3.0 * 0.15
        score += metadata_score
        
        # Enhanced categories and keywords diversity bonus
        categories = result.get("categories", [])
        keywords = result.get("filter_keywords", [])
        related_terms = result.get("related_terms", [])
        
        diversity_bonus = 0.0
        if len(categories) >= 3:
            diversity_bonus += 0.05
        if len(keywords) >= 3:
            diversity_bonus += 0.05
        if len(related_terms) >= 3:
            diversity_bonus += 0.05
        
        score += diversity_bonus
        
        # Collaboration bonus for merged results
        # If we detect this is a merged result (multiple diverse sources), add bonus
        if len(categories) >= 4 and len(keywords) >= 4:
            # This indicates merged content from multiple providers
            collaboration_bonus = 0.2
            score += collaboration_bonus
        
        return min(score, 1.0)
    
    def _merge_query_results(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from two providers for optimal coverage."""
        if not result1:
            return result2 or {}
        if not result2:
            return result1
        
        merged = result1.copy()
        
        # Merge queries (remove duplicates, keep best)
        queries1 = result1.get("search_queries", [])
        queries2 = result2.get("search_queries", [])
        
        seen_queries = set()
        merged_queries = []
        
        # Add all queries from result1
        for query in queries1:
            query_text = query.get("query", "")
            if query_text not in seen_queries:
                merged_queries.append(query)
                seen_queries.add(query_text)
        
        # Add unique queries from result2
        for query in queries2:
            query_text = query.get("query", "")
            if query_text not in seen_queries:
                merged_queries.append(query)
                seen_queries.add(query_text)
        
        merged["search_queries"] = merged_queries
        
        # Merge categories, keywords, and related terms
        for field in ["categories", "filter_keywords", "related_terms"]:
            items1 = set(result1.get(field, []))
            items2 = set(result2.get(field, []))
            merged[field] = list(items1.union(items2))
        
        return merged
    
    def _estimate_cost(self, provider: str, topic: str, max_queries: int) -> float:
        """Estimate cost for query generation."""
        # Rough token estimates
        input_tokens = len(topic.split()) * 50 + max_queries * 10  # Prompt tokens
        output_tokens = max_queries * 30  # Generated queries
        
        provider_info = self.provider_factory.get_provider_info(provider)
        cost_per_1k = provider_info["cost_per_1k_tokens"]
        
        input_cost = (input_tokens / 1000) * cost_per_1k["input"]
        output_cost = (output_tokens / 1000) * cost_per_1k["output"]
        
        return input_cost + output_cost
    
    def _update_usage_stats(self, result: CollaborationResult):
        """Update usage statistics."""
        self.usage_stats[result.primary_provider]["requests"] += 1
        self.usage_stats[result.primary_provider]["total_cost"] += result.cost_estimate
        
        if result.secondary_provider:
            self.usage_stats[result.secondary_provider]["requests"] += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        total_requests = sum(stats["requests"] for stats in self.usage_stats.values())
        total_cost = sum(stats["total_cost"] for stats in self.usage_stats.values())
        
        return {
            "providers": self.usage_stats,
            "total_requests": total_requests,
            "total_cost": total_cost,
            "budget_used_percentage": (total_cost / config.openai_budget_limit) * 100,
            "cost_savings": self._calculate_cost_savings()
        }
    
    def _calculate_cost_savings(self) -> Dict[str, float]:
        """Calculate cost savings from using Gemini vs OpenAI only."""
        gemini_requests = self.usage_stats["gemini"]["requests"]
        gemini_cost = self.usage_stats["gemini"]["total_cost"]
        
        # Estimate what it would cost with OpenAI only
        openai_info = self.provider_factory.get_provider_info("openai")
        gemini_info = self.provider_factory.get_provider_info("gemini")
        
        cost_ratio = (
            (openai_info["cost_per_1k_tokens"]["input"] + openai_info["cost_per_1k_tokens"]["output"]) /
            (gemini_info["cost_per_1k_tokens"]["input"] + gemini_info["cost_per_1k_tokens"]["output"])
        )
        
        estimated_openai_cost = gemini_cost * cost_ratio
        savings = estimated_openai_cost - gemini_cost
        
        return {
            "absolute_savings": savings,
            "percentage_savings": (savings / estimated_openai_cost) * 100 if estimated_openai_cost > 0 else 0
        }


# Global collaborative service instance
collaborative_service = CollaborativeService()