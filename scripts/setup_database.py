#!/usr/bin/env python3
"""Database setup script for ArXiv Recommendation System."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arxiv_recommendation.database import DatabaseManager
from arxiv_recommendation.config import config


async def setup_database():
    """Initialize the database with all required tables and indexes."""
    
    print("Setting up ArXiv Recommendation System database...")
    print(f"Database path: {config.database_path}")
    
    # Create database manager and initialize
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        print("✅ Database initialized successfully!")
        
        # Get and display database stats
        stats = await db_manager.get_database_stats()
        print("\n📊 Database Statistics:")
        print(f"  Total papers: {stats['total_papers']}")
        print(f"  Processed papers: {stats['processed_papers']}")
        print(f"  Papers with embeddings: {stats['papers_with_embeddings']}")
        print(f"  User ratings: {stats['user_ratings']}")
        print(f"  Papers added last week: {stats['papers_last_week']}")
        
        # Display user preferences
        preferences = await db_manager.get_user_preferences()
        print("\n⚙️ Default User Preferences:")
        for key, value in preferences.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def main():
    """Main entry point."""
    success = asyncio.run(setup_database())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()