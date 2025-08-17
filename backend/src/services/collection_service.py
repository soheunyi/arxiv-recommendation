#!/usr/bin/env python3
"""
Collection Service for ArXiv Recommendation System.

This service provides paper collection functionality that can be used by both
API endpoints and CLI scripts.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from arxiv_client import ArXivClient, PaperMetadata
from database import DatabaseManager
from .query_service import QueryService
from .query_refinement_service import QueryRefinementService
from .provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


class CollectionService:
    """Service for collecting papers on research topics using GPT-generated queries."""
    
    def __init__(self, topic: str):
        """Initialize the collection service."""
        self.topic = topic
        self.db_manager = DatabaseManager()
        self.query_service = QueryService()
        self.query_refinement_service = QueryRefinementService()
        
        # Collection state
        self.collected_papers: List[PaperMetadata] = []
        self.paper_ids: Set[str] = set()
        self.query_config: Optional[Dict] = None
        
        # Statistics
        self.stats = {
            "queries_executed": 0,
            "queries_refined": 0,
            "papers_found": 0,
            "papers_filtered": 0,
            "papers_deduplicated": 0,
            "refinement_successful": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def generate_queries(self, max_queries: int = 15, use_config: Optional[str] = None, llm_provider: Optional[str] = None) -> Dict:
        """Generate or load search queries for the topic."""
        logger.info(f"Generating search strategy for: '{self.topic}'")
        
        if use_config:
            # Load from configuration file
            logger.info(f"Loading queries from: {use_config}")
            self.query_config = self.query_service.load_queries_config(use_config)
            if not self.query_config:
                raise ValueError(f"Failed to load configuration from {use_config}")
        else:
            # Generate new queries with specified LLM provider
            if llm_provider:
                logger.info(f"Consulting {llm_provider.title()} for optimal search queries...")
                query_service = ProviderFactory.create_query_service(llm_provider)
            else:
                logger.info("Consulting configured LLM provider for optimal search queries...")
                query_service = self.query_service
                
            self.query_config = query_service.generate_search_queries(
                self.topic, max_queries
            )
        
        logger.info(f"Generated {len(self.query_config['search_queries'])} search queries")
        logger.info(f"Target categories: {', '.join(self.query_config['categories'])}")
        
        return self.query_config
    
    def preview_collection_plan(self) -> Dict:
        """Preview the collection plan without executing."""
        if not self.query_config:
            raise ValueError("Must generate queries first")
        
        # Group queries by priority
        high_priority = [q for q in self.query_config['search_queries'] if q['priority'] == 'high']
        medium_priority = [q for q in self.query_config['search_queries'] if q['priority'] == 'medium']
        low_priority = [q for q in self.query_config['search_queries'] if q['priority'] == 'low']
        
        return {
            "topic": self.topic,
            "high_priority_queries": high_priority,
            "medium_priority_queries": medium_priority,
            "low_priority_queries": low_priority,
            "estimated_papers": len(self.query_config['search_queries']) * 15,
            "estimated_time_seconds": len(self.query_config['search_queries']) * 4,
            "categories": self.query_config['categories'],
            "filter_keywords": self.query_config['filter_keywords']
        }
    
    async def search_by_query(self, query_info: Dict, max_results: int = 20) -> List[PaperMetadata]:
        """Execute a single search query with intelligent refinement for poor results."""
        query = query_info['query']
        priority = query_info['priority']
        
        logger.debug(f"Executing {priority.upper()} priority query: {query}")
        
        try:
            client = ArXivClient()
            papers = await client.search_papers(query, max_results=max_results)
            
            logger.debug(f"Query returned {len(papers)} papers")
            
            # Check if query refinement is needed (poor results)
            min_threshold = 3 if priority == 'high' else 1  # Higher expectations for high-priority queries
            
            if self.query_refinement_service.should_refine(papers, min_threshold):
                logger.info(f"🔄 Query returned only {len(papers)} papers (< {min_threshold}), attempting refinement...")
                self.stats["queries_refined"] += 1
                
                # Generate refined query variations
                refinement_result = await self.query_refinement_service.refine_query(
                    original_query=query,
                    topic_context=self.topic,
                    max_variations=3
                )
                
                # Test all refined queries and pick the one with most results
                query_results = []
                
                # Start with original query as baseline
                query_results.append({
                    'query': query,
                    'strategy': 'original',
                    'papers': papers,
                    'count': len(papers),
                    'confidence': 1.0
                })
                
                # Test all refined queries
                for refined_query_data in refinement_result.get('refined_queries', []):
                    refined_query = refined_query_data['query']
                    strategy = refined_query_data['strategy']
                    confidence = refined_query_data['confidence']
                    
                    logger.debug(f"Testing refined query ({strategy}): {refined_query}")
                    
                    try:
                        refined_papers = await client.search_papers(refined_query, max_results=max_results)
                        logger.debug(f"Refined query returned {len(refined_papers)} papers")
                        
                        query_results.append({
                            'query': refined_query,
                            'strategy': strategy,
                            'papers': refined_papers,
                            'count': len(refined_papers),
                            'confidence': confidence
                        })
                        
                    except Exception as e:
                        logger.warning(f"Refined query failed: {e}")
                        continue
                
                # Find the query with most results
                best_result = max(query_results, key=lambda x: x['count'])
                best_papers = best_result['papers']
                
                # Log results summary
                logger.info(f"📊 Query refinement results:")
                for result in sorted(query_results, key=lambda x: x['count'], reverse=True):
                    marker = "🏆" if result == best_result else "📄"
                    logger.info(f"   {marker} {result['strategy']}: {result['count']} papers (confidence: {result['confidence']:.2f})")
                
                # Mark as successful if we found a better query
                if best_result['strategy'] != 'original':
                    logger.info(f"✅ Best refinement: {best_result['strategy']} with {best_result['count']} papers")
                    self.stats["refinement_successful"] += 1
                else:
                    logger.info(f"📝 Original query performed best with {best_result['count']} papers")
                
                # Log refinement outcome
                if len(best_papers) > len(papers):
                    logger.info(f"🎯 Final result: {len(papers)} → {len(best_papers)} papers after refinement")
                else:
                    logger.info(f"📝 Refinement attempted but kept original {len(papers)} papers")
                
                # If still no results after refinement, try DuckDuckGo fallback
                if len(best_papers) == 0:
                    logger.warning(f"🔄 All refinement failed (0 results), trying DuckDuckGo fallback: {query}")
                    ddg_papers = await self._attempt_duckduckgo_fallback(query, client, max_results)
                    if ddg_papers:
                        return ddg_papers
                
                return best_papers
            
            # If original query returned 0 results and no refinement was triggered, try DuckDuckGo
            if len(papers) == 0:
                logger.warning(f"🔄 Query returned 0 results, trying DuckDuckGo fallback: {query}")
                ddg_papers = await self._attempt_duckduckgo_fallback(query, client, max_results)
                if ddg_papers:
                    return ddg_papers
            
            return papers
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def filter_papers_by_relevance(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Filter papers based on relevance to the topic."""
        if not self.query_config or not self.query_config.get('filter_keywords'):
            return papers
        
        keywords = [k.lower() for k in self.query_config['filter_keywords']]
        filtered_papers = []
        
        for paper in papers:
            # Check if any keyword appears in title or abstract
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower()
            
            if any(keyword in title_lower or keyword in abstract_lower for keyword in keywords):
                filtered_papers.append(paper)
        
        logger.debug(f"Filtered {len(papers)} papers to {len(filtered_papers)} relevant papers")
        return filtered_papers
    
    async def collect_papers(
        self, 
        max_papers_per_query: int = 20, 
        skip_filtering: bool = False
    ) -> List[PaperMetadata]:
        """Execute the full paper collection process."""
        if not self.query_config:
            raise ValueError("Must generate queries first")
        
        logger.info(f"Starting paper collection for '{self.topic}'")
        self.stats["start_time"] = datetime.now()
        
        all_papers = []
        
        # Execute queries in priority order
        queries_by_priority = {
            'high': [q for q in self.query_config['search_queries'] if q['priority'] == 'high'],
            'medium': [q for q in self.query_config['search_queries'] if q['priority'] == 'medium'],
            'low': [q for q in self.query_config['search_queries'] if q['priority'] == 'low']
        }
        
        for priority in ['high', 'medium', 'low']:
            queries = queries_by_priority[priority]
            if not queries:
                continue
                
            logger.info(f"Executing {priority.upper()} priority queries ({len(queries)} queries)")
            
            for i, query_info in enumerate(queries):
                try:
                    # Execute search
                    papers = await self.search_by_query(query_info, max_papers_per_query)
                    self.stats["queries_executed"] += 1
                    self.stats["papers_found"] += len(papers)
                    
                    # Filter for relevance unless skipped
                    if not skip_filtering:
                        papers = self.filter_papers_by_relevance(papers)
                        self.stats["papers_filtered"] += len(papers)
                    
                    # Add to collection, avoiding duplicates
                    new_papers = 0
                    for paper in papers:
                        if paper.id not in self.paper_ids:
                            all_papers.append(paper)
                            self.paper_ids.add(paper.id)
                            new_papers += 1
                        else:
                            self.stats["papers_deduplicated"] += 1
                    
                    logger.debug(f"Added {new_papers} new papers (avoided {len(papers) - new_papers} duplicates)")
                    
                    # Rate limiting - be respectful to arXiv API
                    if i < len(queries) - 1:  # Don't wait after the last query
                        await asyncio.sleep(3.5)
                
                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    continue
        
        self.collected_papers = all_papers
        self.stats["end_time"] = datetime.now()
        
        logger.info(f"Collection complete! Total unique papers: {len(all_papers)}")
        logger.info(f"Queries executed: {self.stats['queries_executed']}")
        logger.info(f"Queries refined: {self.stats['queries_refined']}")
        logger.info(f"Successful refinements: {self.stats['refinement_successful']}")
        logger.info(f"Total papers found: {self.stats['papers_found']}")
        logger.info(f"Duplicates removed: {self.stats['papers_deduplicated']}")
        
        # Calculate refinement success rate
        if self.stats['queries_refined'] > 0:
            success_rate = (self.stats['refinement_successful'] / self.stats['queries_refined']) * 100
            logger.info(f"Refinement success rate: {success_rate:.1f}%")
        
        return all_papers
    
    def get_collection_stats(self) -> Dict:
        """Get detailed collection statistics including query refinement metrics."""
        stats = self.stats.copy()
        
        # Add calculated metrics
        if stats['queries_executed'] > 0:
            stats['refinement_rate'] = (stats['queries_refined'] / stats['queries_executed']) * 100
        else:
            stats['refinement_rate'] = 0.0
            
        if stats['queries_refined'] > 0:
            stats['refinement_success_rate'] = (stats['refinement_successful'] / stats['queries_refined']) * 100
        else:
            stats['refinement_success_rate'] = 0.0
        
        # Add timing if available
        if stats['start_time'] and stats['end_time']:
            duration = stats['end_time'] - stats['start_time']
            stats['duration_seconds'] = duration.total_seconds()
        
        return stats
    
    async def clean_database(self, 
                             backup_metadata: Optional[Dict] = None,
                             granular_options: Optional[Dict] = None) -> Dict:
        """
        Clean the database with enhanced safety and backup features.
        
        Args:
            backup_metadata: Metadata to include in the backup
            granular_options: Options for granular cleaning (keep_ratings, keep_embeddings, etc.)
            
        Returns:
            Dictionary with cleanup results and backup information
        """
        try:
            logger.info("Starting enhanced database cleanup with backup protection...")
            
            # Initialize backup service
            from .backup_service import BackupService
            backup_service = BackupService()
            
            # Create automatic backup before cleaning
            backup_metadata = backup_metadata or {
                "reason": "pre_clean_safety_backup",
                "requested_at": datetime.now().isoformat()
            }
            
            backup_result = await backup_service.create_backup("pre_clean", backup_metadata)
            if not backup_result["success"]:
                raise Exception(f"Failed to create safety backup: {backup_result.get('error')}")
            
            logger.info(f"Safety backup created: {backup_result['metadata']['filename']}")
            
            # Initialize database connection
            await self.db_manager.initialize()
            
            import aiosqlite
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                # Determine what to clean based on granular options
                tables_to_clear = self._get_tables_to_clean(granular_options)
                
                cleaned_counts = {}
                
                for table in tables_to_clear:
                    try:
                        # Get count before deletion
                        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                        count_before = (await cursor.fetchone())[0]
                        
                        # Delete records
                        await db.execute(f"DELETE FROM {table}")
                        cleaned_counts[table] = count_before
                        
                        logger.info(f"Cleaned {table} table: {count_before} records removed")
                        
                    except Exception as e:
                        logger.error(f"Failed to clean {table}: {e}")
                        cleaned_counts[table] = {"error": str(e)}
                
                await db.commit()
            
            logger.info("Database cleaning completed successfully")
            
            return {
                "success": True,
                "backup_info": backup_result["metadata"],
                "cleaned_tables": cleaned_counts,
                "total_records_removed": sum(
                    count for count in cleaned_counts.values() 
                    if isinstance(count, int)
                )
            }
            
        except Exception as e:
            logger.error(f"Database cleaning failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "backup_info": backup_result.get("metadata") if 'backup_result' in locals() else None
            }
    
    def _get_tables_to_clean(self, granular_options: Optional[Dict]) -> List[str]:
        """Determine which tables to clean based on granular options."""
        all_tables = [
            "papers",
            "user_ratings", 
            "paper_embeddings",
            "search_history",
            "recommendations_history"
        ]
        
        if not granular_options:
            return all_tables
        
        tables_to_clean = []
        
        # Add tables based on options
        if granular_options.get("clean_papers", True):
            tables_to_clean.append("papers")
        
        if granular_options.get("clean_ratings", True):
            tables_to_clean.append("user_ratings")
        
        if granular_options.get("clean_embeddings", True):
            tables_to_clean.append("paper_embeddings")
        
        if granular_options.get("clean_search_history", True):
            tables_to_clean.append("search_history")
        
        if granular_options.get("clean_recommendations", True):
            tables_to_clean.append("recommendations_history")
        
        return tables_to_clean
    
    async def store_papers(self, fetch_references: bool = True) -> int:
        """Store collected papers in the database with optional reference fetching.
        
        Args:
            fetch_references: Whether to immediately fetch references (Stage 1)
            
        Returns:
            Number of papers stored
        """
        if not self.collected_papers:
            logger.warning("No papers to store")
            return 0
        
        logger.info(f"Storing {len(self.collected_papers)} papers...")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Store papers
        stored_count = await self.db_manager.store_papers(self.collected_papers)
        
        # Stage 1: Immediate reference fetching from ArXiv HTML
        if fetch_references and stored_count > 0:
            await self._fetch_references_stage1()
        
        logger.info(f"Successfully stored {stored_count} papers")
        return stored_count
    
    def get_collection_summary(self) -> Dict:
        """Get detailed summary of the collection."""
        if not self.collected_papers:
            return {"error": "No papers collected"}
        
        # Count by category
        category_counts = {}
        for paper in self.collected_papers:
            category = paper.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Recent vs older papers
        current_year = datetime.now().year
        recent_papers = [p for p in self.collected_papers if p.published_date.year >= current_year - 1]
        
        # Sample papers
        sample_papers = []
        for paper in self.collected_papers[:5]:
            sample_papers.append({
                "title": paper.title,
                "authors": paper.authors[:2] + (["..."] if len(paper.authors) > 2 else []),
                "category": paper.category,
                "published_date": paper.published_date.date().isoformat(),
                "arxiv_url": paper.arxiv_url
            })
        
        return {
            "topic": self.topic,
            "total_papers": len(self.collected_papers),
            "category_counts": category_counts,
            "recent_papers_count": len(recent_papers),
            "older_papers_count": len(self.collected_papers) - len(recent_papers),
            "sample_papers": sample_papers,
            "statistics": self.stats
        }
    
    def save_collection_report(self, filepath: str) -> bool:
        """Save a detailed collection report."""
        try:
            report = {
                "topic": self.topic,
                "collection_timestamp": datetime.now().isoformat(),
                "query_config": self.query_config,
                "statistics": self.stats,
                "papers_collected": len(self.collected_papers),
                "papers_summary": [
                    {
                        "id": paper.id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "category": paper.category,
                        "published_date": paper.published_date.isoformat(),
                        "arxiv_url": paper.arxiv_url
                    }
                    for paper in self.collected_papers
                ]
            }
            
            report_path = Path(filepath)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Collection report saved to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save collection report: {e}")
            return False
    
    def save_config(self, filepath: str) -> bool:
        """Save the current query configuration to a file."""
        if not self.query_config:
            logger.error("No query configuration to save")
            return False
        
        return self.query_service.save_queries_config(self.query_config, filepath)
    
    async def _fetch_references_stage1(self) -> None:
        """Fetch references for collected papers using Stage 1 (ArXiv HTML parsing)."""
        if not self.collected_papers:
            return
            
        logger.info(f"Starting Stage 1 reference fetching for {len(self.collected_papers)} papers...")
        
        try:
            from .hybrid_reference_service import HybridReferenceService
            
            # Initialize hybrid service
            hybrid_service = HybridReferenceService()
            
            # Track progress
            processed = 0
            successful = 0
            
            for paper in self.collected_papers:
                try:
                    # Fetch Stage 1 references (ArXiv HTML parsing)
                    references = await hybrid_service.fetch_references_stage1(paper.id)
                    
                    if references:
                        successful += 1
                        logger.debug(f"Stage 1: Found {len(references)} references for {paper.id}")
                    else:
                        logger.debug(f"Stage 1: No references found for {paper.id}")
                    
                    processed += 1
                    
                    # Rate limiting to be respectful
                    if processed < len(self.collected_papers):
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    logger.error(f"Stage 1 reference fetching failed for {paper.id}: {e}")
                    processed += 1
                    continue
            
            logger.info(f"Stage 1 reference fetching completed: {successful}/{processed} papers processed successfully")
            logger.info("Stage 2 (OpenAlex enrichment) will be handled automatically by scheduler")
            
        except Exception as e:
            logger.error(f"Failed to initialize reference fetching: {e}")
    
    async def execute_collection(
        self, 
        max_papers_per_query: int = 20,
        max_queries: int = 15,
        skip_filtering: bool = False,
        fetch_references: bool = True,
        filter_recent_days: Optional[int] = None,
        clean_db: bool = False,
        llm_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the complete collection workflow with two-stage reference fetching.
        
        Args:
            max_papers_per_query: Maximum papers per query
            max_queries: Maximum queries to generate
            skip_filtering: Skip relevance filtering
            fetch_references: Enable Stage 1 reference fetching
            filter_recent_days: Only collect papers from last N days
            clean_db: Clean database before collection
            llm_provider: LLM provider for query generation
            
        Returns:
            Dict with collection results and statistics
        """
        workflow_start = datetime.now()
        
        try:
            logger.info(f"Starting complete collection workflow for topic: '{self.topic}'")
            
            # Step 1: Optional database cleaning
            if clean_db:
                logger.info("Cleaning database before collection...")
                clean_result = await self.clean_database()
                if not clean_result.get("success"):
                    logger.warning(f"Database cleaning failed: {clean_result.get('error')}")
            
            # Step 2: Generate search queries
            await self.generate_queries(max_queries=max_queries, llm_provider=llm_provider)
            
            # Step 3: Collect papers
            papers = await self.collect_papers(
                max_papers_per_query=max_papers_per_query,
                skip_filtering=skip_filtering
            )
            
            # Step 4: Apply date filtering if requested
            if filter_recent_days and papers:
                cutoff_date = datetime.now() - timedelta(days=filter_recent_days)
                papers = [p for p in papers if p.published_date >= cutoff_date]
                self.collected_papers = papers  # Update the collection
                logger.info(f"Filtered to {len(papers)} papers from last {filter_recent_days} days")
            
            # Step 5: Store papers with Stage 1 reference fetching
            papers_stored = await self.store_papers(fetch_references=fetch_references)
            
            # Compile results
            workflow_end = datetime.now()
            workflow_duration = (workflow_end - workflow_start).total_seconds()
            
            results = {
                "success": True,
                "topic": self.topic,
                "papers_collected": len(papers),
                "papers_stored": papers_stored,
                "workflow_duration_seconds": workflow_duration,
                "stage1_references_enabled": fetch_references,
                "stage2_scheduled": fetch_references,  # OpenAlex enrichment via scheduler
                "statistics": self.stats,
                "collection_summary": self.get_collection_summary() if papers else None
            }
            
            logger.info(f"Collection workflow completed successfully in {workflow_duration:.1f}s")
            logger.info(f"Papers collected: {len(papers)}, stored: {papers_stored}")
            
            if fetch_references:
                logger.info("Stage 1 (ArXiv) references processed immediately")
                logger.info("Stage 2 (OpenAlex) enrichment will be handled by scheduler")
            
            return results
            
        except Exception as e:
            logger.error(f"Collection workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "topic": self.topic,
                "papers_collected": 0,
                "papers_stored": 0
            }
    
    async def _attempt_duckduckgo_fallback(self, original_query: str, client: ArXivClient, max_results: int) -> List[PaperMetadata]:
        """
        Attempt to find ArXiv papers using DuckDuckGo as a fallback.
        
        This method searches for ArXiv papers that may not have been found through
        direct ArXiv API searches, using enhanced query patterns including "arxiv".
        
        Args:
            original_query: The original query that failed
            client: ArXiv client for fetching paper metadata
            max_results: Maximum number of papers to return
            
        Returns:
            List of PaperMetadata objects found through DuckDuckGo search
        """
        try:
            from .duckduckgo_academic_service import search_arxiv_papers_via_duckduckgo
            
            # Enhanced query patterns including "arxiv" keyword (as suggested by user)
            arxiv_focused_queries = [
                f"{original_query} arxiv",  # User's suggestion: include "arxiv" in query
                f"arxiv {original_query}",
                f'"{original_query}" arxiv.org',
                f"{original_query} preprint arxiv"
            ]
            
            total_arxiv_ids = set()
            
            for i, enhanced_query in enumerate(arxiv_focused_queries):
                try:
                    logger.info(f"🌐 DuckDuckGo search {i+1}/{len(arxiv_focused_queries)}: '{enhanced_query}'")
                    
                    # Search for ArXiv papers via DuckDuckGo
                    arxiv_ids = await search_arxiv_papers_via_duckduckgo(
                        enhanced_query, 
                        max_results=10
                    )
                    
                    if arxiv_ids:
                        logger.info(f"🎯 DuckDuckGo found {len(arxiv_ids)} ArXiv IDs: {arxiv_ids}")
                        total_arxiv_ids.update(arxiv_ids)
                        
                        # Log cumulative progress
                        logger.info(f"📈 Cumulative ArXiv IDs found: {len(total_arxiv_ids)} total")
                        
                        # Don't search more patterns if we found enough results
                        if len(total_arxiv_ids) >= 5:
                            logger.info(f"🎯 Reached target of {len(total_arxiv_ids)} ArXiv papers, stopping DuckDuckGo search")
                            break
                    else:
                        logger.info(f"🌐 No ArXiv papers found for pattern: '{enhanced_query}'")
                        
                except Exception as e:
                    logger.warning(f"🌐 DuckDuckGo search failed for '{enhanced_query}': {e}")
                    continue
            
            if not total_arxiv_ids:
                logger.info(f"🌐 DuckDuckGo fallback complete: No ArXiv papers found for query '{original_query}'")
                logger.info(f"🔍 Searched {len(arxiv_focused_queries)} enhanced query patterns")
                return []
            
            # Fetch full paper metadata using existing ArXiv infrastructure
            logger.info(f"📄 DuckDuckGo fallback successful: Found {len(total_arxiv_ids)} unique ArXiv IDs")
            logger.info(f"📄 ArXiv IDs to fetch: {sorted(list(total_arxiv_ids))}")
            logger.info(f"📄 Fetching metadata for {len(total_arxiv_ids)} ArXiv papers via ArXiv API")
            
            papers = []
            
            for idx, arxiv_id in enumerate(sorted(total_arxiv_ids), 1):
                try:
                    logger.debug(f"📄 Fetching {idx}/{len(total_arxiv_ids)}: {arxiv_id}")
                    
                    # Use existing ArXiv client to get full metadata
                    paper = await client.get_paper_by_id(arxiv_id)
                    
                    if paper:
                        papers.append(paper)
                        logger.debug(f"✅ New paper added: {paper.title[:50]}...")
                        
                        # Rate limiting for ArXiv API (use collection service rate limit)
                        await asyncio.sleep(0.5)  # Lighter rate limiting for metadata fetch
                    else:
                        logger.warning(f"📄 ArXiv API couldn't find paper: {arxiv_id}")
                        
                except Exception as e:
                    logger.warning(f"📄 Failed to fetch ArXiv paper {arxiv_id}: {e}")
                    continue
            
            if papers:
                logger.info(f"✅ DuckDuckGo fallback complete: {len(papers)} papers retrieved")
                logger.info(f"📊 DuckDuckGo fallback summary:")
                logger.info(f"   - Original query: '{original_query}'")
                logger.info(f"   - Enhanced patterns tried: {len(arxiv_focused_queries)}")
                logger.info(f"   - ArXiv IDs found: {len(total_arxiv_ids)}")
                logger.info(f"   - Papers retrieved: {len(papers)}")
                
                return papers[:max_results]  # Limit to requested max results
            else:
                logger.info(f"🌐 DuckDuckGo found {len(total_arxiv_ids)} ArXiv IDs but failed to retrieve metadata")
                return []
                
        except ImportError:
            logger.warning("🌐 DuckDuckGo service not available (missing ddgs library)")
            return []
        except Exception as e:
            logger.error(f"🌐 DuckDuckGo fallback failed: {e}")
            return []