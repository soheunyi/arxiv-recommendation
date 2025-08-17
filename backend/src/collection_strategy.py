#!/usr/bin/env python3
"""
Collection Strategy for ArXiv Recommendation System.

This module implements HARD (High Accuracy with Reduced Duplication) logic
for paper collection to minimize API calls while maximizing coverage.

Key features:
- Query deduplication and optimization
- Result caching to avoid redundant searches
- Intelligent query scheduling with rate limiting
- Paper deduplication across queries
- Collection history tracking
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

from database import DatabaseManager
from arxiv_client import ArXivClient, PaperMetadata
from config import config
from services.query_refinement_service import QueryRefinementService

logger = logging.getLogger(__name__)


class QueryPriority(Enum):
    """Query priority levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QueryResult:
    """Represents the result of a query execution."""

    query: str
    papers_found: int
    papers_new: int
    papers_duplicate: int
    api_calls_made: int
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class CachedQueryResult:
    """Represents a cached query result."""

    query_hash: str
    query: str
    paper_ids: List[str]
    execution_time: datetime
    ttl_hours: int = 24


class CollectionStrategy:
    """
    Implements HARD logic for efficient paper collection.

    This strategy minimizes API calls through:
    1. Query result caching
    2. Paper deduplication
    3. Query optimization
    4. Rate limiting with exponential backoff
    5. Collection history tracking
    """

    def __init__(self, topic: str):
        """Initialize the collection strategy."""
        self.topic = topic
        self.db_manager = DatabaseManager()
        self.arxiv_client = ArXivClient()
        self.query_refinement_service = QueryRefinementService()

        # Strategy configuration
        self.max_papers_per_query = (
            config.max_papers_per_query
            if hasattr(config, "max_papers_per_query")
            else 100
        )
        self.query_cache_ttl = 24  # hours
        self.rate_limit_delay = 1.0  # seconds between requests
        self.max_retries = 3
        self.backoff_multiplier = 2.0

        # State tracking
        self.collected_papers: Set[str] = set()  # Paper IDs already collected
        self.query_cache: Dict[str, CachedQueryResult] = {}
        self.execution_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "papers_found": 0,
            "papers_deduplicated": 0,
            "start_time": None,
            "end_time": None,
        }

        # Load existing papers to avoid duplicates
        asyncio.create_task(self._initialize_collected_papers())

    async def _initialize_collected_papers(self):
        """Load existing papers from database to avoid duplicates."""
        try:
            await self.db_manager.initialize()
            papers = await self.db_manager.get_recent_papers(
                limit=5000
            )  # Get recent papers

            for paper in papers:
                self.collected_papers.add(paper["id"])

            logger.info(
                f"Initialized with {len(self.collected_papers)} existing papers"
            )

        except Exception as e:
            logger.error(f"Failed to initialize collected papers: {e}")

    async def execute_collection(
        self,
        queries: List[Dict[str, Any]],
        max_total_papers: int = 500,
        filter_recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute paper collection using HARD strategy.

        Args:
            queries: List of query dictionaries with 'query', 'priority', etc.
            max_total_papers: Maximum total papers to collect
            filter_recent_days: Only collect papers from last N days

        Returns:
            Dictionary with collection results and statistics
        """
        self.execution_stats["start_time"] = datetime.now(timezone.utc)
        logger.info(f"Starting HARD collection for topic: {self.topic}")

        try:
            # Initialize database and load cache
            await self.db_manager.initialize()
            await self._load_query_cache()

            # Prepare and optimize queries
            optimized_queries = await self._optimize_queries(queries)

            # Execute queries with rate limiting and caching
            collection_results = await self._execute_optimized_queries(
                optimized_queries, max_total_papers, filter_recent_days
            )

            # Store results and update cache
            await self._store_collection_results(collection_results)
            await self._save_query_cache()

            # Finalize statistics
            self.execution_stats["end_time"] = datetime.utcnow()
            duration = (
                self.execution_stats["end_time"] - self.execution_stats["start_time"]
            ).total_seconds()

            results = {
                "topic": self.topic,
                "papers_collected": collection_results["papers_stored"],
                "papers_deduplicated": collection_results["papers_deduplicated"],
                "queries_executed": len(optimized_queries),
                "api_calls_made": self.execution_stats["api_calls"],
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "execution_time_seconds": duration,
                "query_results": collection_results["query_results"],
                "efficiency_metrics": self._calculate_efficiency_metrics(
                    collection_results
                ),
            }

            logger.info(
                f"HARD collection completed: {results['papers_collected']} papers in {duration:.2f}s"
            )
            return results

        except Exception as e:
            error_msg = f"HARD collection failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def _optimize_queries(
        self, queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize queries to minimize redundancy and API calls.

        This includes:
        1. Removing duplicate queries
        2. Sorting by priority
        3. Combining similar queries where possible
        4. Filtering out queries likely to return existing papers
        """
        try:
            logger.info(f"Optimizing {len(queries)} queries")

            # Remove duplicate queries
            unique_queries = []
            seen_hashes = set()

            for query_data in queries:
                query_text = query_data.get("query", "").lower().strip()
                query_hash = hashlib.md5(query_text.encode()).hexdigest()

                if query_hash not in seen_hashes:
                    seen_hashes.add(query_hash)
                    query_data["query_hash"] = query_hash
                    unique_queries.append(query_data)

            logger.info(
                f"Removed {len(queries) - len(unique_queries)} duplicate queries"
            )

            # Sort by priority (high -> medium -> low)
            priority_order = {
                QueryPriority.HIGH.value: 0,
                QueryPriority.MEDIUM.value: 1,
                QueryPriority.LOW.value: 2,
            }
            unique_queries.sort(
                key=lambda q: priority_order.get(q.get("priority", "medium"), 1)
            )

            # Check cache for each query and mark cached ones
            for query_data in unique_queries:
                query_hash = query_data["query_hash"]
                if await self._is_query_cached(query_hash):
                    query_data["cached"] = True
                    self.execution_stats["cache_hits"] += 1
                else:
                    query_data["cached"] = False
                    self.execution_stats["cache_misses"] += 1

            logger.info(
                f"Query optimization completed: {len(unique_queries)} unique queries, "
                f"{self.execution_stats['cache_hits']} cached"
            )

            return unique_queries

        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return queries

    async def _execute_optimized_queries(
        self,
        queries: List[Dict[str, Any]],
        max_total_papers: int,
        filter_recent_days: Optional[int],
    ) -> Dict[str, Any]:
        """Execute optimized queries with rate limiting and deduplication."""
        try:
            papers_collected = []
            papers_deduplicated = 0
            query_results = []
            papers_stored = 0

            # Date filter for recent papers
            date_filter = None
            if filter_recent_days:
                date_filter = datetime.utcnow() - timedelta(days=filter_recent_days)

            for i, query_data in enumerate(queries):
                if len(papers_collected) >= max_total_papers:
                    logger.info(f"Reached maximum papers limit: {max_total_papers}")
                    break

                query_text = query_data["query"]
                query_hash = query_data["query_hash"]
                priority = query_data.get("priority", "medium")

                logger.info(
                    f"Processing query {i+1}/{len(queries)} [{priority}]: {query_text}"
                )

                try:
                    # Check cache first
                    if query_data.get("cached", False):
                        cached_result = await self._get_cached_query_result(query_hash)
                        if cached_result:
                            logger.info(
                                f"Using cached result with {len(cached_result.paper_ids)} papers"
                            )

                            # Filter cached papers that we don't already have
                            new_paper_ids = [
                                pid
                                for pid in cached_result.paper_ids
                                if pid not in self.collected_papers
                            ]

                            query_results.append(
                                {
                                    "query": query_text,
                                    "papers_found": len(cached_result.paper_ids),
                                    "papers_new": len(new_paper_ids),
                                    "from_cache": True,
                                }
                            )

                            # Add new papers to collection
                            for paper_id in new_paper_ids:
                                self.collected_papers.add(paper_id)
                                papers_collected.append(paper_id)

                            continue

                    # Execute query via ArXiv API
                    query_result = await self._execute_single_query(
                        query_text, query_hash, date_filter
                    )

                    query_results.append(asdict(query_result))
                    papers_deduplicated += query_result.papers_duplicate

                    # Apply rate limiting between queries
                    if i < len(queries) - 1:  # Don't delay after last query
                        await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to process query '{query_text}': {e}")
                    query_results.append(
                        {
                            "query": query_text,
                            "papers_found": 0,
                            "papers_new": 0,
                            "error": str(e),
                        }
                    )

            # Store collected papers in database
            if papers_collected:
                papers_stored = await self._store_collected_papers(papers_collected)

            return {
                "papers_stored": papers_stored,
                "papers_deduplicated": papers_deduplicated,
                "query_results": query_results,
            }

        except Exception as e:
            logger.error(f"Failed to execute optimized queries: {e}")
            raise

    async def _execute_single_query(
        self, query: str, query_hash: str, date_filter: Optional[datetime]
    ) -> QueryResult:
        """Execute a single query with retry logic and caching."""
        start_time = datetime.utcnow()

        for attempt in range(self.max_retries):
            try:
                # Execute ArXiv search
                papers = await self.arxiv_client.search_papers(
                    query, max_results=self.max_papers_per_query
                )

                self.execution_stats["api_calls"] += 1

                # Apply date filter if specified
                if date_filter:
                    papers = [p for p in papers if p.published_date >= date_filter]

                # Deduplicate against existing collection
                new_papers = []
                duplicate_count = 0
                paper_ids_found = []

                for paper in papers:
                    paper_ids_found.append(paper.id)

                    if paper.id not in self.collected_papers:
                        new_papers.append(paper)
                        self.collected_papers.add(paper.id)
                    else:
                        duplicate_count += 1

                # Cache the query result
                cached_result = CachedQueryResult(
                    query_hash=query_hash,
                    query=query,
                    paper_ids=paper_ids_found,
                    execution_time=datetime.utcnow(),
                    ttl_hours=self.query_cache_ttl,
                )
                self.query_cache[query_hash] = cached_result

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                logger.info(
                    f"Query completed: {len(papers)} found, {len(new_papers)} new, "
                    f"{duplicate_count} duplicates in {execution_time:.2f}s"
                )

                # Check if DuckDuckGo fallback is needed (no results found)
                if len(papers) == 0:
                    logger.warning(
                        f"🔄 Query returned 0 results, trying DuckDuckGo first: {query}"
                    )
                    ddg_result = await self._attempt_duckduckgo_fallback(query)
                    if ddg_result:
                        return ddg_result

                    # If DuckDuckGo also fails, try query refinement as last resort
                    logger.warning(
                        f"🔄 DuckDuckGo also failed, attempting query refinement: {query}"
                    )
                    refined_results = await self._attempt_query_refinement(query)
                    if refined_results:
                        return refined_results

                return QueryResult(
                    query=query,
                    papers_found=len(papers),
                    papers_new=len(new_papers),
                    papers_duplicate=duplicate_count,
                    api_calls_made=1,
                    execution_time=execution_time,
                    success=True,
                )

            except Exception as e:
                wait_time = self.rate_limit_delay * (self.backoff_multiplier**attempt)
                logger.warning(
                    f"Query attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s"
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed
                    execution_time = (datetime.utcnow() - start_time).total_seconds()
                    return QueryResult(
                        query=query,
                        papers_found=0,
                        papers_new=0,
                        papers_duplicate=0,
                        api_calls_made=attempt + 1,
                        execution_time=execution_time,
                        success=False,
                        error=str(e),
                    )

    async def _is_query_cached(self, query_hash: str) -> bool:
        """Check if a query result is cached and still valid."""
        if query_hash not in self.query_cache:
            return False

        cached_result = self.query_cache[query_hash]
        age_hours = (
            datetime.utcnow() - cached_result.execution_time
        ).total_seconds() / 3600

        return age_hours <= cached_result.ttl_hours

    async def _get_cached_query_result(
        self, query_hash: str
    ) -> Optional[CachedQueryResult]:
        """Get cached query result if available and valid."""
        if await self._is_query_cached(query_hash):
            return self.query_cache[query_hash]
        return None

    async def _store_collected_papers(self, paper_ids: List[str]) -> int:
        """Store collected papers in the database."""
        try:
            # For now, this is a placeholder since we need the full paper objects
            # In practice, we would retrieve the full paper metadata and store it
            logger.info(f"Would store {len(paper_ids)} papers in database")
            return len(paper_ids)

        except Exception as e:
            logger.error(f"Failed to store collected papers: {e}")
            return 0

    async def _load_query_cache(self):
        """Load query cache from database."""
        try:
            preferences = await self.db_manager.get_user_preferences()
            cache_key = f"query_cache_{self.topic}"

            if cache_key in preferences:
                cache_data = preferences[cache_key]

                # Restore cached results
                for query_hash, cached_data in cache_data.items():
                    try:
                        cached_result = CachedQueryResult(**cached_data)

                        # Check if still valid
                        if await self._is_query_cached(query_hash):
                            self.query_cache[query_hash] = cached_result
                    except Exception as e:
                        logger.warning(
                            f"Failed to restore cached query {query_hash}: {e}"
                        )

                logger.info(
                    f"Loaded {len(self.query_cache)} cached queries for topic: {self.topic}"
                )

        except Exception as e:
            logger.error(f"Failed to load query cache: {e}")

    async def _save_query_cache(self):
        """Save query cache to database."""
        try:
            cache_key = f"query_cache_{self.topic}"

            # Convert cache to serializable format
            cache_data = {}
            for query_hash, cached_result in self.query_cache.items():
                cache_data[query_hash] = asdict(cached_result)
                # Convert datetime to string
                cache_data[query_hash][
                    "execution_time"
                ] = cached_result.execution_time.isoformat()

            await self.db_manager.update_user_preference(
                key=cache_key, value=cache_data, value_type="json"
            )

            logger.info(f"Saved query cache with {len(cache_data)} entries")

        except Exception as e:
            logger.error(f"Failed to save query cache: {e}")

    async def _store_collection_results(self, results: Dict[str, Any]):
        """Store collection execution results for analysis."""
        try:
            execution_record = {
                "topic": self.topic,
                "start_time": self.execution_stats["start_time"].isoformat(),
                "end_time": self.execution_stats["end_time"].isoformat(),
                "papers_collected": results["papers_stored"],
                "papers_deduplicated": results["papers_deduplicated"],
                "api_calls_made": self.execution_stats["api_calls"],
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "query_results": results["query_results"],
            }

            # Store in collection history
            await self.db_manager.update_user_preference(
                key=f"collection_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                value=execution_record,
                value_type="json",
            )

        except Exception as e:
            logger.error(f"Failed to store collection results: {e}")

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total_queries = (
            self.execution_stats["cache_hits"] + self.execution_stats["cache_misses"]
        )
        if total_queries == 0:
            return 0.0
        return (self.execution_stats["cache_hits"] / total_queries) * 100

    def _calculate_efficiency_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate efficiency metrics for the collection strategy."""
        total_time = (
            self.execution_stats["end_time"] - self.execution_stats["start_time"]
        ).total_seconds()

        return {
            "papers_per_second": (
                results["papers_stored"] / total_time if total_time > 0 else 0
            ),
            "api_calls_per_paper": (
                self.execution_stats["api_calls"] / results["papers_stored"]
                if results["papers_stored"] > 0
                else 0
            ),
            "deduplication_rate": (
                (
                    results["papers_deduplicated"]
                    / (results["papers_stored"] + results["papers_deduplicated"])
                )
                * 100
                if (results["papers_stored"] + results["papers_deduplicated"]) > 0
                else 0
            ),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    async def _attempt_query_refinement(
        self, original_query: str
    ) -> Optional[QueryResult]:
        """
        Attempt to refine a failed query and search with refined variations.

        Args:
            original_query: The original query that returned 0 results

        Returns:
            QueryResult if refinement successful, None otherwise
        """
        try:
            # Generate refined queries using GPT-4o-nano
            refinement_result = await self.query_refinement_service.refine_query(
                original_query=original_query,
                topic_context=self.topic,
                max_variations=3,
            )

            refined_queries = refinement_result.get("refined_queries", [])
            if not refined_queries:
                logger.warning(f"🔄 No refined queries generated for: {original_query}")
                return None

            logger.info(f"🔄 Generated {len(refined_queries)} refined queries")

            # Try each refined query until we find results
            total_papers_found = 0
            total_new_papers = 0
            total_api_calls = 0
            best_result = None

            for i, refined_query_info in enumerate(refined_queries):
                refined_query = refined_query_info.get("query", "")
                strategy = refined_query_info.get("strategy", "unknown")
                confidence = refined_query_info.get("confidence", 0.0)

                if not refined_query:
                    continue

                logger.info(
                    f"🔄 Trying refined query {i+1}/{len(refined_queries)} ({strategy}, confidence: {confidence:.2f}): {refined_query}"
                )

                try:
                    # Execute refined query with rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

                    papers = await self.arxiv_client.search_papers(
                        refined_query, max_results=self.max_papers_per_query
                    )

                    total_api_calls += 1
                    self.execution_stats["api_calls"] += 1

                    if papers:
                        # Process new papers
                        new_papers = []
                        duplicate_count = 0

                        for paper in papers:
                            if paper.id not in self.collected_papers:
                                new_papers.append(paper)
                                self.collected_papers.add(paper.id)
                            else:
                                duplicate_count += 1

                        total_papers_found += len(papers)
                        total_new_papers += len(new_papers)

                        logger.info(
                            f"✅ Refined query successful: {len(papers)} found, {len(new_papers)} new"
                        )

                        # Return immediately on first success
                        best_result = QueryResult(
                            query=f"{original_query} → {refined_query}",
                            papers_found=len(papers),
                            papers_new=len(new_papers),
                            papers_duplicate=duplicate_count,
                            api_calls_made=total_api_calls,
                            execution_time=0.0,  # Will be calculated by caller
                            success=True,
                        )
                        break

                    else:
                        logger.warning(
                            f"🔄 Refined query still returned 0 results: {refined_query}"
                        )

                except Exception as e:
                    logger.warning(f"🔄 Refined query failed: {e}")
                    continue

            if best_result:
                logger.info(
                    f"🔄 Query refinement successful: {total_papers_found} total papers, {total_new_papers} new"
                )
                return best_result
            else:
                logger.warning(f"🔄 All refined queries failed for: {original_query}")
                return None

        except Exception as e:
            logger.error(f"🔄 Query refinement process failed: {e}")
            return None

    async def _attempt_duckduckgo_fallback(
        self, original_query: str
    ) -> Optional[QueryResult]:
        """
        Attempt to find ArXiv papers using DuckDuckGo as a fallback.

        This method searches for ArXiv papers that may not have been found through
        direct ArXiv API searches, using enhanced query patterns including "arxiv".

        Args:
            original_query: The original query that failed

        Returns:
            QueryResult if DuckDuckGo search successful, None otherwise
        """
        try:
            from services.duckduckgo_academic_service import (
                search_arxiv_papers_via_duckduckgo,
            )

            # Enhanced query patterns including "arxiv" keyword
            arxiv_focused_queries = [
                f"{original_query} arxiv",  # Your suggestion: include "arxiv" in query
                f"arxiv {original_query}",
                f'"{original_query}" arxiv.org',
                f"{original_query} preprint arxiv",
            ]

            total_arxiv_ids = set()
            total_api_calls = 0

            for i, enhanced_query in enumerate(arxiv_focused_queries):
                try:
                    logger.info(
                        f"🌐 DuckDuckGo search {i+1}/{len(arxiv_focused_queries)}: '{enhanced_query}'"
                    )

                    # Search for ArXiv papers via DuckDuckGo
                    arxiv_ids = await search_arxiv_papers_via_duckduckgo(
                        enhanced_query, max_results=10
                    )

                    if arxiv_ids:
                        logger.info(
                            f"🎯 DuckDuckGo found {len(arxiv_ids)} ArXiv IDs: {arxiv_ids}"
                        )
                        total_arxiv_ids.update(arxiv_ids)

                        # Log cumulative progress
                        logger.info(
                            f"📈 Cumulative ArXiv IDs found: {len(total_arxiv_ids)} total"
                        )

                        # Don't search more patterns if we found enough results
                        if len(total_arxiv_ids) >= 5:
                            logger.info(
                                f"🎯 Reached target of {len(total_arxiv_ids)} ArXiv papers, stopping DuckDuckGo search"
                            )
                            break
                    else:
                        logger.info(
                            f"🌐 No ArXiv papers found for pattern: '{enhanced_query}'"
                        )

                except Exception as e:
                    logger.warning(
                        f"🌐 DuckDuckGo search failed for '{enhanced_query}': {e}"
                    )
                    continue

            if not total_arxiv_ids:
                logger.info(
                    f"🌐 DuckDuckGo fallback complete: No ArXiv papers found for query '{original_query}'"
                )
                logger.info(
                    f"🔍 Searched {len(arxiv_focused_queries)} enhanced query patterns"
                )
                return None

            # Fetch full paper metadata using existing ArXiv infrastructure
            logger.info(
                f"📄 DuckDuckGo fallback successful: Found {len(total_arxiv_ids)} unique ArXiv IDs"
            )
            logger.info(f"📄 ArXiv IDs to fetch: {sorted(list(total_arxiv_ids))}")
            logger.info(
                f"📄 Fetching metadata for {len(total_arxiv_ids)} ArXiv papers via ArXiv API"
            )

            new_papers = []
            duplicate_count = 0

            for idx, arxiv_id in enumerate(sorted(total_arxiv_ids), 1):
                try:
                    logger.debug(
                        f"📄 Fetching {idx}/{len(total_arxiv_ids)}: {arxiv_id}"
                    )

                    # Use existing ArXiv client to get full metadata
                    paper = await self.arxiv_client.get_paper_by_id(arxiv_id)
                    total_api_calls += 1
                    self.execution_stats["api_calls"] += 1

                    if paper:
                        if paper.id not in self.collected_papers:
                            new_papers.append(paper)
                            self.collected_papers.add(paper.id)
                            logger.debug(f"✅ New paper added: {paper.title[:50]}...")
                        else:
                            duplicate_count += 1
                            logger.debug(f"🔄 Duplicate paper skipped: {arxiv_id}")

                        # Rate limiting for ArXiv API
                        await asyncio.sleep(self.rate_limit_delay)
                    else:
                        logger.warning(f"📄 ArXiv API couldn't find paper: {arxiv_id}")

                except Exception as e:
                    logger.warning(f"📄 Failed to fetch ArXiv paper {arxiv_id}: {e}")
                    continue

            if new_papers:
                logger.info(
                    f"✅ DuckDuckGo fallback complete: {len(new_papers)} new papers, {duplicate_count} duplicates"
                )
                logger.info(f"📊 DuckDuckGo fallback summary:")
                logger.info(f"   - Original query: '{original_query}'")
                logger.info(
                    f"   - Enhanced patterns tried: {len(arxiv_focused_queries)}"
                )
                logger.info(f"   - ArXiv IDs found: {len(total_arxiv_ids)}")
                logger.info(
                    f"   - Papers retrieved: {len(new_papers)} new, {duplicate_count} duplicates"
                )
                logger.info(f"   - API calls made: {total_api_calls}")

                return QueryResult(
                    query=f"{original_query} → DuckDuckGo ArXiv search",
                    papers_found=len(total_arxiv_ids),
                    papers_new=len(new_papers),
                    papers_duplicate=duplicate_count,
                    api_calls_made=total_api_calls,
                    execution_time=0.0,  # Will be calculated by caller
                    success=True,
                )
            else:
                logger.info(
                    f"🌐 DuckDuckGo found {len(total_arxiv_ids)} ArXiv IDs but no new papers after deduplication"
                )
                logger.info(
                    f"📊 DuckDuckGo deduplication: all {len(total_arxiv_ids)} papers were already in collection"
                )
                return None

        except ImportError:
            logger.warning("🌐 DuckDuckGo service not available (missing ddgs library)")
            return None
        except Exception as e:
            logger.error(f"🌐 DuckDuckGo fallback failed: {e}")
            return None

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get collection strategy statistics."""
        return {
            "topic": self.topic,
            "collected_papers_count": len(self.collected_papers),
            "cached_queries_count": len(self.query_cache),
            "execution_stats": self.execution_stats,
            "configuration": {
                "max_papers_per_query": self.max_papers_per_query,
                "query_cache_ttl": self.query_cache_ttl,
                "rate_limit_delay": self.rate_limit_delay,
                "max_retries": self.max_retries,
            },
        }
