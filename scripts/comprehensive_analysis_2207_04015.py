#!/usr/bin/env python3
"""
Comprehensive analysis of paper 2207.04015 using all available PyAlex data.
Since the paper has no references indexed, we'll analyze related works and citations.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))


async def comprehensive_analysis():
    """Comprehensive analysis using all available PyAlex data."""
    import os
    
    try:
        import pyalex
        from pyalex import Works
        
        # Configure email if available
        email = os.getenv("OPENALEX_EMAIL")
        if email:
            pyalex.config.email = email
        
        # The OpenAlex ID for paper 2207.04015
        openalex_id = "https://openalex.org/W4285044933"
        
        print(f"📊 COMPREHENSIVE ANALYSIS: ArXiv 2207.04015")
        print("=" * 65)
        print(f"OpenAlex ID: {openalex_id}")
        print("=" * 65)
        
        # Step 1: Get the main paper
        print("📄 PAPER OVERVIEW")
        print("-" * 30)
        work = Works()[openalex_id]
        
        print(f"Title: {work.get('title')}")
        print(f"Authors: {len(work.get('authorships', []))} authors")
        print(f"Year: {work.get('publication_year')}")
        print(f"Type: {work.get('type')}")
        print(f"Language: {work.get('language')}")
        print(f"Citations received: {work.get('cited_by_count', 0)}")
        print(f"References cited: {len(work.get('referenced_works', []))}")
        print(f"Related works: {len(work.get('related_works', []))}")
        
        # Step 2: Topics and Keywords
        topics = work.get('topics', [])
        keywords = work.get('keywords', [])
        
        print(f"\n🔍 RESEARCH TOPICS & KEYWORDS")
        print("-" * 35)
        
        if topics:
            print("Topics:")
            for i, topic in enumerate(topics[:5], 1):
                display_name = topic.get('display_name', 'Unknown')
                score = topic.get('score', 0)
                print(f"  {i}. {display_name} (score: {score:.3f})")
        
        if keywords:
            print(f"\nKeywords:")
            for i, keyword in enumerate(keywords[:10], 1):
                display_name = keyword.get('display_name', 'Unknown')
                score = keyword.get('score', 0)
                print(f"  {i}. {display_name} (score: {score:.3f})")
        
        # Step 3: Related Works Analysis
        related_works = work.get('related_works', [])
        if related_works:
            print(f"\n🔗 RELATED WORKS ANALYSIS")
            print("-" * 30)
            print(f"Found {len(related_works)} related works")
            
            for i, related_id in enumerate(related_works[:5], 1):
                try:
                    related_work = Works()[related_id]
                    if related_work:
                        title = related_work.get('title', 'Unknown Title')
                        year = related_work.get('publication_year', 'Unknown')
                        citations = related_work.get('cited_by_count', 0)
                        print(f"\n{i}. {title}")
                        print(f"   Year: {year}, Citations: {citations}")
                        
                        # Show author overlap
                        related_authors = [auth.get('author', {}).get('display_name', '') 
                                         for auth in related_work.get('authorships', [])]
                        main_authors = [auth.get('author', {}).get('display_name', '') 
                                      for auth in work.get('authorships', [])]
                        
                        author_overlap = set(main_authors) & set(related_authors)
                        if author_overlap:
                            print(f"   Author overlap: {', '.join(author_overlap)}")
                except Exception as e:
                    print(f"{i}. Error retrieving related work: {e}")
        
        # Step 4: Papers that cite this work
        print(f"\n📈 CITATION ANALYSIS")
        print("-" * 25)
        
        # Use the cited_by_api_url to get citing papers
        cited_by_url = work.get('cited_by_api_url')
        if cited_by_url:
            try:
                # Get papers that cite this work
                citing_works = Works().filter(cites=openalex_id).get()
                
                print(f"Papers citing this work: {len(citing_works)}")
                
                if citing_works:
                    print("\nCiting papers:")
                    for i, citing_work in enumerate(citing_works[:3], 1):
                        title = citing_work.get('title', 'Unknown Title')
                        year = citing_work.get('publication_year', 'Unknown')
                        citations = citing_work.get('cited_by_count', 0)
                        print(f"\n{i}. {title}")
                        print(f"   Year: {year}, Citations: {citations}")
                else:
                    print("No citing papers found (may be too recent)")
            except Exception as e:
                print(f"Error retrieving citing papers: {e}")
        
        # Step 5: Publication and Access Information
        print(f"\n📚 PUBLICATION INFO")
        print("-" * 25)
        
        primary_location = work.get('primary_location', {})
        if primary_location:
            source = primary_location.get('source', {})
            print(f"Source: {source.get('display_name', 'Unknown')}")
            print(f"Publisher: {source.get('host_organization_name', 'Unknown')}")
        
        open_access = work.get('open_access', {})
        if open_access:
            print(f"Open Access: {open_access.get('is_oa', False)}")
            print(f"OA URL: {open_access.get('oa_url', 'None')}")
        
        # Step 6: Abstract (if available)
        abstract_index = work.get('abstract_inverted_index', {})
        if abstract_index:
            print(f"\n📝 ABSTRACT ANALYSIS")
            print("-" * 25)
            
            # Reconstruct abstract from inverted index
            word_positions = []
            for word, positions in abstract_index.items():
                for pos in positions:
                    word_positions.append((pos, word))
            
            # Sort by position and take first 50 words
            word_positions.sort()
            abstract_words = [word for pos, word in word_positions[:50]]
            abstract_preview = ' '.join(abstract_words)
            
            print(f"Abstract preview: {abstract_preview}...")
            print(f"Total words in abstract: {len(word_positions)}")
        
        # Step 7: Research Impact Metrics
        print(f"\n📊 IMPACT METRICS")
        print("-" * 20)
        
        fwci = work.get('fwci')
        if fwci:
            print(f"Field-Weighted Citation Impact: {fwci:.3f}")
        
        percentile = work.get('cited_by_percentile_year', {})
        if percentile:
            min_p = percentile.get('min', 0)
            max_p = percentile.get('max', 0)
            print(f"Citation Percentile: {min_p}-{max_p}%")
        
        # Summary
        print(f"\n🎯 ANALYSIS SUMMARY")
        print("=" * 25)
        print(f"• This is a mathematics optimization paper (category: math.OC)")
        print(f"• Published in 2022, currently has {work.get('cited_by_count', 0)} citations")
        print(f"• No reference list indexed in OpenAlex (common for some papers)")
        print(f"• Has {len(related_works)} related works for context")
        print(f"• Research focuses on convergence analysis and optimization algorithms")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the comprehensive analysis."""
    print("🔬 SYSTEMATIC PAPER ANALYSIS USING PYALEX")
    print("ArXiv 2207.04015: Convergence Analyses of Davis-Yin Splitting")
    print("=" * 70)
    
    await comprehensive_analysis()


if __name__ == "__main__":
    asyncio.run(main())