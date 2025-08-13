#!/usr/bin/env python3
"""
Collection Service for ArXiv Recommendation System.

This service provides paper collection functionality that can be used by both
API endpoints and CLI scripts.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from ..arxiv_client import ArXivClient, PaperMetadata
from ..database import DatabaseManager
from .query_service import QueryService

logger = logging.getLogger(__name__)


class CollectionService:
    """Service for collecting papers on research topics using GPT-generated queries."""
    
    def __init__(self, topic: str):
        """Initialize the collection service."""
        self.topic = topic
        self.db_manager = DatabaseManager()
        self.query_service = QueryService()
        
        # Collection state
        self.collected_papers: List[PaperMetadata] = []
        self.paper_ids: Set[str] = set()
        self.query_config: Optional[Dict] = None
        
        # Statistics
        self.stats = {
            "queries_executed": 0,
            "papers_found": 0,
            "papers_filtered": 0,
            "papers_deduplicated": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def generate_queries(self, max_queries: int = 15, use_config: Optional[str] = None) -> Dict:
        """Generate or load search queries for the topic."""
        logger.info(f"Generating search strategy for: '{self.topic}'")
        
        if use_config:
            # Load from configuration file
            logger.info(f"Loading queries from: {use_config}")
            self.query_config = self.query_service.load_queries_config(use_config)
            if not self.query_config:
                raise ValueError(f"Failed to load configuration from {use_config}")
        else:
            # Generate new queries with GPT
            logger.info("Consulting GPT for optimal search queries...")
            self.query_config = self.query_service.generate_search_queries(
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
        """Execute a single search query."""
        query = query_info['query']
        priority = query_info['priority']
        
        logger.debug(f"Executing {priority.upper()} priority query: {query}")
        
        try:
            client = ArXivClient()
            papers = await client.search_papers(query, max_results=max_results)
            
            logger.debug(f"Query returned {len(papers)} papers")
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
        logger.info(f"Total papers found: {self.stats['papers_found']}")
        logger.info(f"Duplicates removed: {self.stats['papers_deduplicated']}")
        
        return all_papers
    
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
    
    async def store_papers(self) -> int:
        """Store collected papers in the database."""
        if not self.collected_papers:
            logger.warning("No papers to store")
            return 0
        
        logger.info(f"Storing {len(self.collected_papers)} papers...")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Store papers
        stored_count = await self.db_manager.store_papers(self.collected_papers)
        
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