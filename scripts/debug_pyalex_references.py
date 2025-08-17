#!/usr/bin/env python3
"""
Debug PyAlex reference retrieval for paper 2207.04015.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "src"))


async def debug_references():
    """Debug PyAlex reference retrieval step by step."""
    import os
    
    try:
        import pyalex
        from pyalex import Works
        
        # Configure email if available
        email = os.getenv("OPENALEX_EMAIL")
        if email:
            pyalex.config.email = email
        
        # The OpenAlex ID we found for paper 2207.04015
        openalex_id = "https://openalex.org/W4285044933"
        
        print(f"🔍 Debugging PyAlex references for: {openalex_id}")
        print("=" * 60)
        
        # Step 1: Get the work directly
        print("📄 Step 1: Getting work from OpenAlex...")
        work = Works()[openalex_id]
        
        if not work:
            print("❌ Could not retrieve work")
            return
        
        print(f"✅ Work retrieved:")
        print(f"   Title: {work.get('title', 'Unknown')}")
        print(f"   Type: {type(work)}")
        print(f"   Keys available: {list(work.keys()) if hasattr(work, 'keys') else 'Not a dict'}")
        
        # Step 2: Check referenced_works
        print(f"\n📚 Step 2: Checking referenced_works...")
        referenced_works = work.get('referenced_works', [])
        print(f"   Referenced works: {type(referenced_works)}")
        print(f"   Count: {len(referenced_works) if referenced_works else 0}")
        
        if referenced_works:
            print(f"   First few reference IDs:")
            for i, ref_id in enumerate(referenced_works[:5]):
                print(f"     {i+1}. {ref_id}")
        else:
            print("   ❌ No referenced works found")
            print("   This could mean:")
            print("     - The paper has no references in OpenAlex")
            print("     - The references aren't indexed yet")
            print("     - There's an issue with data access")
        
        # Step 3: Try to get one reference if available
        if referenced_works:
            print(f"\n🔍 Step 3: Testing reference retrieval...")
            try:
                first_ref_id = referenced_works[0]
                print(f"   Attempting to get: {first_ref_id}")
                
                ref_work = Works()[first_ref_id]
                if ref_work:
                    print(f"   ✅ Reference retrieved:")
                    print(f"     Title: {ref_work.get('title', 'Unknown')}")
                    print(f"     Year: {ref_work.get('publication_year', 'Unknown')}")
                    print(f"     Authors: {len(ref_work.get('authorships', []))} authorships")
                else:
                    print(f"   ❌ Could not retrieve reference")
            except Exception as e:
                print(f"   ❌ Error retrieving reference: {e}")
        
        # Step 4: Check the work structure more deeply
        print(f"\n🔍 Step 4: Deep work structure analysis...")
        for key in ['referenced_works', 'related_works', 'cited_by_count']:
            value = work.get(key)
            print(f"   {key}: {type(value)} = {value}")
        
        # Step 5: Try alternative approaches
        print(f"\n🔄 Step 5: Alternative data access...")
        
        # Try getting the work data in raw format
        print(f"   Raw work data sample:")
        for key, value in list(work.items())[:10]:
            print(f"     {key}: {type(value)}")
    
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run the debug analysis."""
    print("🐛 PYALEX REFERENCE DEBUG")
    print("Investigating why reference retrieval fails")
    print("=" * 50)
    
    await debug_references()


if __name__ == "__main__":
    asyncio.run(main())