#!/usr/bin/env python3
"""
Systematic citation analysis for ArXiv paper 2207.04015 using PyAlex.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))

from services.openalex_service import OpenAlexService
from arxiv_client import ArXivClient


async def analyze_paper_citations(arxiv_id: str):
    """Systematically analyze what a paper cited using PyAlex."""
    import os
    
    print(f"📊 SYSTEMATIC CITATION ANALYSIS")
    print(f"ArXiv ID: {arxiv_id}")
    print("=" * 60)
    
    email = os.getenv("OPENALEX_EMAIL")
    openalex_service = OpenAlexService(email=email)
    
    try:
        # Step 1: Get the paper from ArXiv
        print("📄 Step 1: Fetching paper from ArXiv...")
        async with ArXivClient() as arxiv_client:
            arxiv_paper = await arxiv_client.get_paper_by_id(arxiv_id)
        
        if not arxiv_paper:
            print(f"❌ Could not fetch ArXiv paper {arxiv_id}")
            return
        
        print(f"✅ ArXiv Paper Found:")
        print(f"   Title: {arxiv_paper.title}")
        print(f"   Authors: {', '.join(arxiv_paper.authors[:3])}{'...' if len(arxiv_paper.authors) > 3 else ''}")
        print(f"   Category: {arxiv_paper.category}")
        print(f"   Published: {arxiv_paper.published_date.strftime('%Y-%m-%d')}")
        
        # Step 2: Find the paper in OpenAlex using our improved search
        print(f"\n🔍 Step 2: Finding paper in OpenAlex...")
        openalex_data = await openalex_service.search_comprehensive(arxiv_paper)
        
        if not openalex_data:
            print(f"❌ Paper not found in OpenAlex")
            return
        
        print(f"✅ Found in OpenAlex:")
        print(f"   Method: {openalex_data.get('search_method', 'unknown')}")
        print(f"   OpenAlex ID: {openalex_data['openalex_id']}")
        print(f"   Citations: {openalex_data['cited_by_count']}")
        
        # Step 3: Get references (what this paper cited)
        print(f"\n📚 Step 3: Analyzing what this paper cited...")
        references = await openalex_service.get_paper_references(openalex_data['openalex_id'])
        
        if not references:
            print(f"❌ No references found or error retrieving them")
            return
        
        print(f"✅ Found {len(references)} references")
        
        # Step 4: Detailed analysis of references
        print(f"\n📋 Step 4: Detailed Reference Analysis")
        print("-" * 60)
        
        # Group references by publication year
        by_year = {}
        by_venue = {}
        arxiv_refs = []
        highly_cited = []
        
        for i, ref in enumerate(references, 1):
            ref_title = ref.get('cited_title', 'Unknown Title')
            ref_authors = ref.get('cited_authors', 'Unknown Authors')
            ref_year = ref.get('cited_year')
            is_arxiv = ref.get('is_arxiv_paper', False)
            
            print(f"\n{i:2d}. {ref_title}")
            print(f"    Authors: {ref_authors}")
            print(f"    Year: {ref_year if ref_year else 'Unknown'}")
            print(f"    ArXiv: {'Yes' if is_arxiv else 'No'}")
            print(f"    Source: {ref.get('source', 'unknown')}")
            
            # Collect statistics
            if ref_year:
                by_year[ref_year] = by_year.get(ref_year, 0) + 1
            
            if is_arxiv:
                arxiv_refs.append(ref)
        
        # Step 5: Summary statistics
        print(f"\n📈 Step 5: Citation Statistics")
        print("=" * 60)
        print(f"Total references analyzed: {len(references)}")
        print(f"ArXiv papers cited: {len(arxiv_refs)}")
        print(f"Published papers cited: {len(references) - len(arxiv_refs)}")
        
        if by_year:
            print(f"\n📅 References by Year:")
            for year in sorted(by_year.keys()):
                print(f"   {year}: {by_year[year]} papers")
            
            oldest_year = min(by_year.keys())
            newest_year = max(by_year.keys())
            print(f"\n   Citation span: {oldest_year} - {newest_year} ({newest_year - oldest_year} years)")
        
        # Step 6: Show ArXiv papers cited
        if arxiv_refs:
            print(f"\n📄 ArXiv Papers Cited:")
            print("-" * 40)
            for i, ref in enumerate(arxiv_refs, 1):
                print(f"{i}. {ref.get('cited_title', 'Unknown')}")
                if ref.get('cited_paper_id'):
                    print(f"   ArXiv ID: {ref['cited_paper_id']}")
                print(f"   Year: {ref.get('cited_year', 'Unknown')}")
        
        # Step 7: Citation network analysis
        print(f"\n🕸️  Step 6: Citation Network Context")
        print("-" * 40)
        
        # Get papers that cite this paper
        try:
            network_data = await openalex_service.get_citation_network(openalex_data['openalex_id'])
            citing_papers = network_data.get('citing_papers', [])
            
            print(f"Papers citing this work: {len(citing_papers)}")
            print(f"Papers this work cites: {len(references)}")
            print(f"Citation impact: {openalex_data['cited_by_count']} total citations")
            
            if citing_papers:
                print(f"\nSample papers citing this work:")
                for i, citing in enumerate(citing_papers[:3], 1):
                    print(f"  {i}. {citing.get('title', 'Unknown')[:60]}...")
                    print(f"     Year: {citing.get('publication_year', 'Unknown')}")
                    print(f"     Citations: {citing.get('cited_by_count', 0)}")
        
        except Exception as e:
            print(f"⚠️  Could not retrieve citation network: {e}")
        
        print(f"\n🎯 Analysis Complete!")
        print(f"This provides a comprehensive view of paper {arxiv_id}'s citation behavior.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the systematic citation analysis."""
    arxiv_id = "2207.04015"
    
    print("🔬 COMPREHENSIVE CITATION ANALYSIS")
    print("Using PyAlex and our improved OpenAlex integration")
    print("=" * 55)
    
    await analyze_paper_citations(arxiv_id)


if __name__ == "__main__":
    asyncio.run(main())