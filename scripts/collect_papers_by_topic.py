#!/usr/bin/env python3
"""
Generalized arXiv paper collection script with GPT-powered query generation.

This script can collect papers on any research topic by using GPT to generate
intelligent search queries and then executing them against the arXiv API.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from arxiv_recommendation.services import CollectionService


class TopicPaperCollector:
    """Collect papers on any research topic using GPT-generated queries."""
    
    def __init__(self, topic: str):
        """Initialize collector - now just a wrapper around CollectionService."""
        self.collection_service = CollectionService(topic)
        
        # Expose service attributes for backward compatibility
        self.topic = self.collection_service.topic
        self.db_manager = self.collection_service.db_manager
        self.query_service = self.collection_service.query_service
        self.collected_papers = self.collection_service.collected_papers
        self.paper_ids = self.collection_service.paper_ids
        self.query_config = self.collection_service.query_config
        self.stats = self.collection_service.stats

    async def generate_queries(self, max_queries: int = 15, use_config: Optional[str] = None):
        """Generate or load search queries for the topic."""
        print(f"🤖 Generating search strategy for: '{self.topic}'")
        
        result = await self.collection_service.generate_queries(max_queries, use_config)
        
        # Update exposed attributes
        self.query_config = self.collection_service.query_config
        
        print(f"✅ Generated {len(self.query_config['search_queries'])} search queries")
        print(f"🏷️ Target categories: {', '.join(self.query_config['categories'])}")
        
        return result
    
    def preview_collection_plan(self) -> None:
        """Preview the collection plan without executing."""
        if not self.query_config:
            raise ValueError("Must generate queries first")
        
        preview = self.collection_service.preview_collection_plan()
        
        print(f"\n📋 Collection Plan for '{self.topic}'")
        print("=" * 60)
        
        # Display by priority
        for priority, emoji in [("high_priority_queries", "🔴"), ("medium_priority_queries", "🟡"), ("low_priority_queries", "🟢")]:
            queries = preview.get(priority, [])
            if queries:
                priority_name = priority.replace("_queries", "").replace("_", " ").upper()
                print(f"\n{emoji} {priority_name} QUERIES ({len(queries)} queries):")
                for i, query in enumerate(queries, 1):
                    print(f"   {i}. {query['query']}")
                    print(f"      → {query['description']}")
        
        print(f"\n🎯 Estimated collection size: {preview['estimated_papers']} papers")
        print(f"⏱️ Estimated time: {preview['estimated_time_seconds']:.1f} seconds (with rate limiting)")

    async def search_by_query(self, query_info, max_results: int = 20):
        """Execute a single search query."""
        return await self.collection_service.search_by_query(query_info, max_results)

    def filter_papers_by_relevance(self, papers):
        """Filter papers based on relevance to the topic."""
        return self.collection_service.filter_papers_by_relevance(papers)

    async def collect_papers(self, max_papers_per_query: int = 20, skip_filtering: bool = False):
        """Execute the full paper collection process."""
        print(f"🚀 Starting collection for '{self.topic}'")
        
        papers = await self.collection_service.collect_papers(max_papers_per_query, skip_filtering)
        
        # Update exposed attributes
        self.collected_papers = self.collection_service.collected_papers
        self.paper_ids = self.collection_service.paper_ids
        self.stats = self.collection_service.stats
        
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        print(f"   Collection time: {duration:.1f} seconds")
        
        return papers

    async def clean_database(self) -> None:
        """Clean the database while preserving schema and preferences."""
        print("🧹 Cleaning database...")
        await self.collection_service.clean_database()
        print("✅ Database cleaned successfully")

    async def store_papers(self) -> int:
        """Store collected papers in the database."""
        stored_count = await self.collection_service.store_papers()
        return stored_count

    def show_collection_summary(self) -> None:
        """Display detailed summary of the collection."""
        summary = self.collection_service.get_collection_summary()
        
        if "error" in summary:
            print("⚠️ No papers collected")
            return
        
        print(f"\n📊 Collection Summary for '{summary['topic']}':")
        print("=" * 60)
        
        print(f"\n📚 Papers by arXiv category:")
        for category, count in sorted(summary['category_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {category}: {count} papers")
        
        print(f"\n📅 Paper timeline:")
        print(f"   Recent papers (last 2 years): {summary['recent_papers_count']}")
        print(f"   Older papers: {summary['older_papers_count']}")
        
        print(f"\n📄 Sample papers:")
        for i, paper in enumerate(summary['sample_papers'], 1):
            print(f"   {i}. {paper['title'][:70]}...")
            print(f"      Authors: {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
            print(f"      Category: {paper['category']} | Date: {paper['published_date']}")
            print()
        
        if summary['total_papers'] > 5:
            print(f"   ... and {summary['total_papers'] - 5} more papers")

    def save_collection_report(self, filepath: str) -> None:
        """Save a detailed collection report."""
        success = self.collection_service.save_collection_report(filepath)
        if success:
            print(f"📝 Collection report saved to: {filepath}")
        else:
            print(f"❌ Failed to save collection report to: {filepath}")


async def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect arXiv papers on any research topic using GPT-generated queries"
    )
    
    # Required arguments
    parser.add_argument("topic", help="Research topic to collect papers for")
    
    # Collection options
    parser.add_argument("--max-papers", type=int, default=20,
                       help="Maximum papers per query (default: 20)")
    parser.add_argument("--max-queries", type=int, default=15,
                       help="Maximum queries to generate (default: 15)")
    
    # Database options
    parser.add_argument("--clean-db", action="store_true",
                       help="Clean database before collection")
    parser.add_argument("--no-clean", action="store_true",
                       help="Don't clean database (append mode)")
    
    # Configuration options
    parser.add_argument("--config", type=str,
                       help="Load queries from configuration file")
    parser.add_argument("--save-config", type=str,
                       help="Save generated queries to configuration file")
    
    # Execution modes
    parser.add_argument("--preview", action="store_true",
                       help="Preview collection plan without executing")
    parser.add_argument("--dry-run", action="store_true",
                       help="Collect papers but don't store in database")
    
    # Output options
    parser.add_argument("--save-report", type=str,
                       help="Save detailed collection report to file")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip relevance filtering of papers")
    
    args = parser.parse_args()
    
    try:
        # Initialize collector
        collector = TopicPaperCollector(args.topic)
        
        # Generate queries
        await collector.generate_queries(
            max_queries=args.max_queries,
            use_config=args.config
        )
        
        # Save config if requested
        if args.save_config:
            collector.collection_service.save_config(args.save_config)
            print(f"💾 Saved configuration to: {args.save_config}")
        
        # Preview mode - just show the plan
        if args.preview:
            collector.preview_collection_plan()
            return
        
        # Clean database if requested (and not in no-clean mode)
        if args.clean_db and not args.no_clean:
            await collector.clean_database()
        
        # Collect papers
        papers = await collector.collect_papers(
            max_papers_per_query=args.max_papers,
            skip_filtering=args.skip_filtering
        )
        
        # Store papers unless dry run
        if not args.dry_run:
            stored_count = await collector.store_papers()
            print(f"✅ Successfully stored {stored_count} papers")
        else:
            print(f"🔍 Dry run complete - found {len(papers)} papers (not stored)")
        
        # Show summary
        collector.show_collection_summary()
        
        # Save report if requested
        if args.save_report:
            collector.save_collection_report(args.save_report)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())