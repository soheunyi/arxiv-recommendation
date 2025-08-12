#!/usr/bin/env python3
"""
Fix for ArXiv Recommendation System Rating Update Button

This script provides a simplified, reliable approach to update paper ratings
that fixes the async/sync integration issues.
"""

def update_paper_rating_fixed(paper_id: str, rating: int, notes: str = None):
    """Simplified synchronous rating update function."""
    import asyncio
    import streamlit as st
    import logging
    
    if not st.session_state.db_manager:
        st.error("Database not initialized")
        return False
    
    try:
        # Simple approach: Use asyncio.run() in a clean context
        success = asyncio.run(
            st.session_state.db_manager.store_user_rating(
                paper_id=paper_id,
                rating=rating,
                notes=notes
            )
        )
        
        if success:
            st.success(f"Rating {rating}/5 updated successfully!")
            # Clear any relevant caches
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            # Trigger rerun to refresh UI
            st.rerun()
            return True
        else:
            st.error("Failed to update rating")
            return False
            
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Fallback: Use synchronous SQLite
            return update_rating_sync_fallback(paper_id, rating, notes)
        else:
            st.error(f"Runtime error: {e}")
            logging.exception(f"Rating update failed for paper {paper_id}")
            return False
            
    except Exception as e:
        st.error(f"Error updating rating: {e}")
        logging.exception(f"Rating update failed for paper {paper_id}")
        return False

def update_rating_sync_fallback(paper_id: str, rating: int, notes: str = None):
    """Fallback synchronous rating update using regular SQLite."""
    import sqlite3
    import streamlit as st
    from datetime import datetime
    import logging
    
    try:
        db_path = st.session_state.db_manager.db_path
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_ratings (paper_id, rating, notes, updated_at)
                VALUES (?, ?, ?, ?)
            """, (paper_id, rating, notes, datetime.now()))
            
            conn.commit()
            
        st.success(f"Rating {rating}/5 updated successfully!")
        logging.info(f"Stored rating {rating} for paper {paper_id} (sync fallback)")
        
        # Clear caches and rerun
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):  
            st.cache_resource.clear()
        st.rerun()
        return True
        
    except Exception as e:
        st.error(f"Fallback rating update failed: {e}")
        logging.exception(f"Sync fallback failed for paper {paper_id}")
        return False

# Alternative: Completely async approach
def show_dashboard_fixed():
    """Fixed dashboard with proper async handling."""
    import streamlit as st
    import asyncio
    
    st.title("📚 ArXiv Recommendation Dashboard")
    
    # Use async approach with proper error handling
    if st.button("🔄 Load Recommendations"):
        try:
            # Run async operation properly
            papers = asyncio.run(get_recent_recommendations_safe())
            st.session_state['dashboard_papers'] = papers
        except Exception as e:
            st.error(f"Failed to load recommendations: {e}")
            st.session_state['dashboard_papers'] = []
    
    # Display papers from session state
    papers = st.session_state.get('dashboard_papers', [])
    
    if papers:
        for i, paper in enumerate(papers[:10], 1):
            with st.container():
                st.markdown(f"**#{i}. {paper.get('title', 'Unknown')}**")
                
                # Simplified rating interface
                current_rating = paper.get('rating', 0)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    new_rating = st.slider(
                        f"Rate paper {i}",
                        min_value=0,
                        max_value=5,
                        value=current_rating,
                        key=f"rating_fixed_{paper['id']}"
                    )
                
                with col2:
                    if st.button("Update", key=f"update_fixed_{paper['id']}"):
                        # Use the fixed update function
                        success = update_paper_rating_fixed(paper['id'], new_rating)
                        if success:
                            # Update local state immediately
                            paper['rating'] = new_rating
                
                st.divider()

async def get_recent_recommendations_safe():
    """Safe async function to get recommendations."""
    import streamlit as st
    
    if not st.session_state.db_manager:
        return []
    
    try:
        papers = await st.session_state.db_manager.get_recent_papers(limit=10)
        return papers
    except Exception as e:
        st.error(f"Database error: {e}")
        return []

if __name__ == "__main__":
    print("Rating button fix functions defined.")
    print("To apply fix:")
    print("1. Replace update_paper_rating_sync() with update_paper_rating_fixed()")
    print("2. Update button click handlers to use the new function")
    print("3. Consider using show_dashboard_fixed() as well")