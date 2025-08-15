#!/usr/bin/env python3
"""
Generalized arXiv paper collection script with GPT-powered query generation.

This script can collect papers on any research topic by using GPT to generate
intelligent search queries and then executing them against the arXiv API.
"""

import asyncio
import sys

from script_utils import (
    setup_backend_path, print_success, print_error, print_progress, print_info
)

# Setup backend imports
setup_backend_path()
from services import CollectionService


def display_collection_plan(service: CollectionService) -> None:
    """Display the collection plan preview."""
    if not service.query_config:
        raise ValueError("Must generate queries first")
    
    preview = service.preview_collection_plan()
    
    print(f"\n📋 Collection Plan for '{service.topic}'")
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


def display_collection_summary(service: CollectionService) -> None:
    """Display detailed summary of the collection."""
    summary = service.get_collection_summary()
    
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
        # Initialize collection service
        service = CollectionService(args.topic)
        
        # Generate queries
        print_progress(f"Generating search strategy for: '{args.topic}'")
        await service.generate_queries(
            max_queries=args.max_queries,
            use_config=args.config
        )
        print_success(f"Generated {len(service.query_config['search_queries'])} search queries")
        print_info(f"Target categories: {', '.join(service.query_config['categories'])}")
        
        # Save config if requested
        if args.save_config:
            service.save_config(args.save_config)
            print(f"💾 Saved configuration to: {args.save_config}")
        
        # Preview mode - just show the plan
        if args.preview:
            display_collection_plan(service)
            return
        
        # Clean database if requested (and not in no-clean mode)
        if args.clean_db and not args.no_clean:
            print_progress("Cleaning database...")
            await service.clean_database()
            print_success("Database cleaned successfully")
        
        # Collect papers
        print_progress(f"Starting collection for '{args.topic}'")
        papers = await service.collect_papers(
            max_papers_per_query=args.max_papers,
            skip_filtering=args.skip_filtering
        )
        
        duration = (service.stats["end_time"] - service.stats["start_time"]).total_seconds()
        print_info(f"Collection time: {duration:.1f} seconds")
        
        # Store papers unless dry run
        if not args.dry_run:
            stored_count = await service.store_papers()
            print_success(f"Successfully stored {stored_count} papers")
        else:
            print_info(f"Dry run complete - found {len(papers)} papers (not stored)")
        
        # Show summary
        display_collection_summary(service)
        
        # Save report if requested
        if args.save_report:
            success = service.save_collection_report(args.save_report)
            if success:
                print_success(f"Collection report saved to: {args.save_report}")
            else:
                print_error(f"Failed to save collection report to: {args.save_report}")
            
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())