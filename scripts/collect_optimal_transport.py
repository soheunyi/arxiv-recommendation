#!/usr/bin/env python3
"""
Optimal Transport Paper Collection Script

This script collects papers related to optimal transport from arXiv,
cleans the database, and populates it with OT-focused research.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from arxiv_client import ArXivClient
from database import DatabaseManager
from config import config


class OptimalTransportCollector:
    """Collector for optimal transport papers from arXiv."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.collected_papers = []
        
        # Optimal transport related search queries with proper arXiv API syntax
        self.search_queries = [
            # Core optimal transport terms
            'all:"optimal transport"',
            'all:"optimal transportation"', 
            'ti:"optimal transport" OR abs:"optimal transport"',
            
            # Wasserstein distance variants
            'all:"wasserstein distance"',
            'all:"wasserstein metric"',
            'all:"wasserstein space"',
            'ti:"wasserstein" OR abs:"wasserstein"',
            
            # Earth mover distance
            'all:"earth mover distance"',
            'all:"EMD"',
            
            # Classical optimal transport
            'all:"kantorovich problem"',
            'all:"monge problem"',
            'all:"transportation problem"',
            'all:"mass transport"',
            
            # Algorithmic approaches
            'all:"sinkhorn algorithm"',
            'all:"entropic regularization"',
            'abs:"sinkhorn" AND abs:"optimal"',
            
            # Barycenter and geometry
            'all:"wasserstein barycenter"',
            'all:"optimal coupling"',
            'abs:"barycenter" AND abs:"wasserstein"',
            
            # Broader OT applications
            'ti:"optimal transport" AND cat:cs.LG',
            'ti:"wasserstein" AND cat:stat.ML',
            'abs:"optimal transport" AND abs:"machine learning"'
        ]
        
        self.target_categories = [
            "stat.OT",  # Statistics - Other Statistics (includes OT)
            "math.OC",  # Optimization and Control
            "cs.LG",    # Machine Learning (many OT applications)
            "stat.ML",  # Machine Learning (statistics)
            "math.PR",  # Probability Theory
            "math.AP",  # Analysis of PDEs
            "cs.CV",    # Computer Vision (OT applications)
            "econ.TH",  # Economic Theory (transportation economics)
        ]
    
    async def clean_database(self):
        """Clean the database while preserving schema and preferences."""
        print("🧹 Cleaning database tables...")
        
        await self.db_manager.initialize()
        
        # Get connection and clear tables
        import aiosqlite
        async with aiosqlite.connect(self.db_manager.db_path) as db:
            # Clear all data tables but keep schema
            tables_to_clear = [
                "papers",
                "user_ratings", 
                "paper_embeddings",
                "search_history",
                "recommendations_history"
            ]
            
            for table in tables_to_clear:
                await db.execute(f"DELETE FROM {table}")
                print(f"   ✅ Cleared {table} table")
            
            await db.commit()
        
        print("✅ Database cleaned successfully")
    
    async def search_by_query(self, query: str, max_results: int = 50) -> list:
        """Search for papers using a specific query."""
        print(f"🔍 Searching for: '{query}'")
        
        async with ArXivClient() as client:
            papers = await client.search_papers(
                query=query,
                max_results=max_results
            )
            
            print(f"   Found {len(papers)} papers")
            return papers
    
    async def search_by_category(self, category: str, max_results: int = 30) -> list:
        """Search for recent papers in a specific category."""
        print(f"🏷️ Searching category: {category}")
        
        async with ArXivClient() as client:
            # Get papers from last 30 days to ensure fresh content
            papers = await client.fetch_recent_papers(
                category=category,
                max_results=max_results,
                days_back=30
            )
            
            print(f"   Found {len(papers)} papers")
            return papers
    
    async def filter_optimal_transport_papers(self, papers: list) -> list:
        """Filter papers to keep only those relevant to optimal transport."""
        filtered_papers = []
        
        # Keywords that indicate optimal transport relevance
        ot_keywords = [
            "optimal transport", "wasserstein", "earth mover", "transportation",
            "kantorovich", "monge", "sinkhorn", "entropic regularization",
            "mass transport", "optimal coupling", "barycenter", "displacement"
        ]
        
        for paper in papers:
            # Check title and abstract for OT keywords
            text_to_check = (paper.title + " " + paper.abstract).lower()
            
            # Check if any OT keyword appears
            has_ot_keyword = any(keyword in text_to_check for keyword in ot_keywords)
            
            if has_ot_keyword:
                filtered_papers.append(paper)
        
        print(f"   📋 Filtered to {len(filtered_papers)} OT-relevant papers")
        return filtered_papers
    
    async def collect_all_papers(self):
        """Main collection process for optimal transport papers."""
        print("🚀 Starting optimal transport paper collection...")
        
        all_papers = []
        paper_ids = set()  # To avoid duplicates
        
        # 1. Search by specific queries
        print("\n📝 Phase 1: Searching by optimal transport queries...")
        for i, query in enumerate(self.search_queries):
            try:
                print(f"   Query {i+1}/{len(self.search_queries)}: {query[:50]}...")
                papers = await self.search_by_query(query, max_results=20)
                
                # Filter for relevance and remove duplicates
                filtered_papers = await self.filter_optimal_transport_papers(papers)
                
                for paper in filtered_papers:
                    if paper.id not in paper_ids:
                        all_papers.append(paper)
                        paper_ids.add(paper.id)
                
                # Rate limiting - be respectful to arXiv API
                await asyncio.sleep(3.5)
                
            except Exception as e:
                print(f"   ⚠️ Error searching for '{query}': {e}")
        
        print(f"\n📊 After query search: {len(all_papers)} unique papers")
        
        # 2. Search by categories
        print("\n🏷️ Phase 2: Searching by relevant categories...")
        for category in self.target_categories:
            try:
                papers = await self.search_by_category(category, max_results=20)
                
                # Filter for OT relevance
                filtered_papers = await self.filter_optimal_transport_papers(papers)
                
                for paper in filtered_papers:
                    if paper.id not in paper_ids:
                        all_papers.append(paper)
                        paper_ids.add(paper.id)
                
                # Rate limiting
                await asyncio.sleep(3.5)
                
            except Exception as e:
                print(f"   ⚠️ Error searching category '{category}': {e}")
        
        print(f"\n📊 Total collected: {len(all_papers)} unique optimal transport papers")
        
        self.collected_papers = all_papers
        return all_papers
    
    async def store_papers(self):
        """Store collected papers in the database."""
        if not self.collected_papers:
            print("⚠️ No papers to store")
            return
        
        print(f"💾 Storing {len(self.collected_papers)} papers...")
        
        # Store papers in database
        stored_count = await self.db_manager.store_papers(self.collected_papers)
        
        print(f"✅ Successfully stored {stored_count} papers")
        
        # Show summary statistics
        await self.show_collection_summary()
    
    async def show_collection_summary(self):
        """Display summary of collected papers."""
        print("\n📊 Collection Summary:")
        
        # Count by category
        category_counts = {}
        for paper in self.collected_papers:
            category = paper.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("   Papers by category:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"     {category}: {count}")
        
        # Show some sample papers
        print("\n📄 Sample papers:")
        for i, paper in enumerate(self.collected_papers[:5]):
            print(f"   {i+1}. {paper.title[:80]}...")
            print(f"      Category: {paper.category}, Date: {paper.published_date.date()}")
        
        if len(self.collected_papers) > 5:
            print(f"   ... and {len(self.collected_papers) - 5} more")
    
    async def run_full_collection(self, clean_db: bool = True):
        """Run the complete collection process."""
        print("🎯 Optimal Transport Paper Collection Starting...")
        print(f"   Target: Comprehensive collection of OT papers")
        print(f"   Clean database: {clean_db}")
        
        try:
            # Initialize database
            await self.db_manager.initialize()
            
            # Clean database if requested
            if clean_db:
                await self.clean_database()
            
            # Collect papers
            await self.collect_all_papers()
            
            # Store papers
            await self.store_papers()
            
            print("\n🎉 Collection complete!")
            return True
            
        except Exception as e:
            print(f"❌ Collection failed: {e}")
            return False


async def main():
    """Main function to run the collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect optimal transport papers")
    parser.add_argument("--no-clean", action="store_true", 
                       help="Don't clean database before collection")
    parser.add_argument("--dry-run", action="store_true",
                       help="Collect papers but don't store them")
    
    args = parser.parse_args()
    
    collector = OptimalTransportCollector()
    
    if args.dry_run:
        print("🧪 DRY RUN MODE - Papers will be collected but not stored")
        await collector.collect_all_papers()
        await collector.show_collection_summary()
    else:
        clean_db = not args.no_clean
        success = await collector.run_full_collection(clean_db=clean_db)
        
        if success:
            print("\n✨ Next steps:")
            print("   1. Restart the API server to see new papers")
            print("   2. Visit frontend to browse optimal transport papers")
            print("   3. Start rating papers to build recommendations")
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())