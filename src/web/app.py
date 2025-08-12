"""
ArXiv Recommendation System - Streamlit Web Interface
"""

import asyncio
from asyncio.log import logger
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arxiv_recommendation.database import DatabaseManager
from arxiv_recommendation.agents import MultiAgentSystem
from arxiv_recommendation.config import config
from arxiv_recommendation.embeddings import EmbeddingManager

# Configure Streamlit page
st.set_page_config(
    page_title="ArXiv Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.db_manager = None
    st.session_state.system = None
    st.session_state.embedding_manager = None

# Minimal CSS - star rating buttons only
st.markdown(
    """
<style>
    .rating-stars {
        color: #ffa500;
        font-size: 1.2em;
    }
    
    /* JavaScript-powered star button styling - only applied to buttons with star emojis */
    .star-rating-button {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0.2rem !important;
        margin: 0.05rem !important;
        font-size: 1.6em !important;
        transition: transform 0.15s ease !important;
        min-height: 2rem !important;
        height: 2rem !important;
        width: 2rem !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        vertical-align: middle !important;
    }
    
    /* Hover states for star buttons only */
    .star-rating-button:hover {
        transform: scale(1.1) !important;
        background: rgba(255, 165, 0, 0.1) !important;
        border: none !important;
        color: #ff8c00 !important;
    }
    
    /* Focus states for accessibility */
    .star-rating-button:focus {
        outline: 2px solid #ff6b6b !important;
        outline-offset: 2px !important;
        border: none !important;
        background: rgba(255, 165, 0, 0.15) !important;
    }
    
    /* Rate label alignment */
    .star-rating-container {
        display: flex !important;
        align-items: center !important;
        gap: 0.25rem !important;
    }
    
    .star-rating-container strong {
        line-height: 2rem !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }
</style>

<script>
// Add class to star buttons for reliable targeting
function markStarButtons() {
    const buttons = document.querySelectorAll('button[data-testid="stBaseButton-secondary"]');
    buttons.forEach(button => {
        const text = button.textContent.trim();
        if (text === '⭐' || text === '☆') {
            button.classList.add('star-rating-button');
        } else {
            button.classList.remove('star-rating-button');
        }
    });
}

// Run when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', markStarButtons);
} else {
    markStarButtons();
}

// Re-mark after DOM changes (Streamlit re-renders)
const observer = new MutationObserver(function(mutations) {
    let shouldRecheck = false;
    mutations.forEach(mutation => {
        if (mutation.addedNodes.length > 0) {
            shouldRecheck = true;
        }
    });
    if (shouldRecheck) {
        setTimeout(markStarButtons, 100); // Small delay to ensure DOM is fully updated
    }
});
observer.observe(document.body, { childList: true, subtree: true });

// Also run periodically as backup
setInterval(markStarButtons, 2000);
</script>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_system(cache_buster: str = "v1-db-current-score"):
    """Initialize the system components."""
    try:
        db_manager = DatabaseManager()
        system = MultiAgentSystem()
        embedding_manager = EmbeddingManager()
        return db_manager, system, embedding_manager
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None


def display_rating_stars(
    rating: int, paper_id: str = None, interactive: bool = False
) -> int:
    """Display rating stars and handle interactions."""
    if interactive and paper_id:
        # Interactive rating widget
        cols = st.columns(5)
        new_rating = rating

        for i in range(1, 6):
            with cols[i - 1]:
                if st.button("⭐" if i <= rating else "☆", key=f"star_{paper_id}_{i}"):
                    new_rating = i
        return new_rating
    else:
        # Display-only stars
        stars = "⭐" * rating + "☆" * (5 - rating)
        return st.markdown(
            f'<span class="rating-stars">{stars}</span>', unsafe_allow_html=True
        )


async def get_papers_with_ratings():
    """Get papers from database with ratings."""
    if not st.session_state.db_manager:
        return []

    try:
        # Get all papers with their ratings
        papers = await st.session_state.db_manager.get_papers_with_ratings()
        return papers
    except Exception as e:
        st.error(f"Error fetching papers: {e}")
        return []


def auto_update_rating(paper_id: str, rating: int, notes: str = None):
    """Automatically update paper rating with simplified approach."""
    if not st.session_state.db_manager:
        return False

    try:
        import asyncio

        success = asyncio.run(
            st.session_state.db_manager.store_user_rating(
                paper_id=paper_id, rating=rating, notes=notes
            )
        )

        if success:
            # Update session state to reflect change immediately
            if "paper_ratings" not in st.session_state:
                st.session_state.paper_ratings = {}
            st.session_state.paper_ratings[paper_id] = rating
            st.success(f"⭐ Rated {rating}/5 stars!", icon="✅")
            return True
        return False

    except Exception as e:
        st.error(f"Rating update failed: {e}")
        return False


def star_rating_widget(
    paper_id: str, current_rating: int = 0, notes: str = None
) -> int:
    """Interactive star rating widget with clickable star display (★★☆☆☆)."""
    # Get current rating from session state if available
    if "paper_ratings" not in st.session_state:
        st.session_state.paper_ratings = {}

    displayed_rating = st.session_state.paper_ratings.get(paper_id, current_rating)

    # Create a very compact row with minimal gap between Rate and stars
    rating_cols = st.columns(
        [0.8, 0.4, 0.4, 0.4, 0.4, 0.4]
    )  # Tight label + 5 compact stars + spacer

    with rating_cols[0]:
        st.markdown("**Rate:**")

    new_rating = displayed_rating

    # Create 5 clickable stars - use minimal button approach
    for star_num in range(1, 6):
        with rating_cols[star_num]:
            star_symbol = "⭐" if star_num <= displayed_rating else "☆"

            # Use Streamlit button with minimal styling
            if st.button(
                star_symbol,
                key=f"star_{paper_id}_{star_num}",
                help=f"Rate {star_num} star{'s' if star_num > 1 else ''}",
                use_container_width=False,  # Don't use full width
            ):
                if auto_update_rating(paper_id, star_num, notes):
                    new_rating = star_num
                    st.rerun()

    return new_rating


def render_latex_abstract(
    abstract_text: str, truncate: bool = False, max_length: int = 200
) -> None:
    """Render abstract text with inline LaTeX support using st.markdown."""
    if not abstract_text:
        st.markdown("*No abstract available*")
        return

    # Handle truncation based on user preference
    display_text = abstract_text
    is_truncated = False
    if truncate and len(abstract_text) > max_length:
        display_text = abstract_text[:max_length] + "..."
        is_truncated = True

    # Check for LaTeX content
    latex_patterns = [
        r"\$\$[^$]+\$\$",  # Display math $$...$$
        r"\$[^$]+\$",  # Inline math $...$
        r"\\[a-zA-Z]+",  # LaTeX commands
    ]

    combined_pattern = "|".join(latex_patterns)
    has_latex = re.search(combined_pattern, display_text)

    try:
        if has_latex:
            # Convert display math $$...$$ to inline $...$ for better flow
            processed_text = re.sub(r"\$\$(.*?)\$\$", r"$\1$", display_text)

            # Render as markdown with LaTeX support
            st.markdown(f"**Abstract:** {processed_text}")
        else:
            # Plain text abstract
            st.markdown(f"**Abstract:** {display_text}")

        if is_truncated:
            st.caption("*(Abstract truncated)*")

    except Exception:
        # Fallback to plain text if any issues
        st.markdown(f"**Abstract:** {display_text}")
        if has_latex:
            st.caption("*(LaTeX rendering simplified due to complexity)*")


async def run_recommendation_workflow():
    """Run the recommendation system workflow."""
    if not st.session_state.system:
        return {"error": "System not initialized"}

    try:
        with st.spinner("Generating recommendations..."):
            from arxiv_recommendation import run_recommendation_system

            results = await run_recommendation_system()
            return results
    except Exception as e:
        return {"error": str(e)}


def main():
    """Main Streamlit application."""

    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            # Pass cache-buster to force new instances when the system changes
            db_manager, system, embedding_manager = initialize_system(
                "v2-db-current-score"
            )
            if db_manager and system and embedding_manager:
                # Ensure database is initialized
                try:
                    asyncio.run(db_manager.initialize())
                except Exception as e:
                    st.warning(
                        f"Database already initialized or initialization issue: {e}"
                    )

                st.session_state.db_manager = db_manager
                st.session_state.system = system
                st.session_state.embedding_manager = embedding_manager
                st.session_state.initialized = True
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system components")
                return

    # Sidebar navigation with buttons
    st.sidebar.title("📚 ArXiv Recommendations")

    # Initialize page selection in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"

    # Navigation buttons in sidebar
    st.sidebar.markdown("### Navigation")

    if st.sidebar.button(
        "🏠 Dashboard",
        type=(
            "primary"
            if st.session_state.current_page == "🏠 Dashboard"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state.current_page = "🏠 Dashboard"
        st.rerun()

    if st.sidebar.button(
        "⭐ Rate Papers",
        type=(
            "primary"
            if st.session_state.current_page == "⭐ Rate Papers"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state.current_page = "⭐ Rate Papers"
        st.rerun()

    if st.sidebar.button(
        "🔍 Search Papers",
        type=(
            "primary"
            if st.session_state.current_page == "🔍 Search Papers"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state.current_page = "🔍 Search Papers"
        st.rerun()

    if st.sidebar.button(
        "⚙️ Settings",
        type=(
            "primary" if st.session_state.current_page == "⚙️ Settings" else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state.current_page = "⚙️ Settings"
        st.rerun()

    if st.sidebar.button(
        "📊 Analytics",
        type=(
            "primary"
            if st.session_state.current_page == "📊 Analytics"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state.current_page = "📊 Analytics"
        st.rerun()

    # Add some spacing and information
    st.sidebar.divider()
    st.sidebar.markdown("### Quick Info")
    st.sidebar.info("💡 Click any button above to navigate between sections")

    # Get current page selection
    page = st.session_state.current_page

    # Main content area
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "⭐ Rate Papers":
        show_rating_interface()
    elif page == "🔍 Search Papers":
        show_search_interface()
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "📊 Analytics":
        show_analytics()


def show_dashboard():
    """Show the main dashboard with recommendations."""
    st.title("📚 ArXiv Recommendation Dashboard")
    st.markdown("Welcome to your personalized research paper recommendation system!")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Generate New Recommendations", type="primary"):
            results = asyncio.run(run_recommendation_workflow())

            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                st.success("Recommendations generated successfully!")
                st.rerun()

    with col2:
        if st.button("📊 View Cache Stats"):
            if st.session_state.embedding_manager:
                stats = st.session_state.embedding_manager.get_cache_stats()

                # Display stats
                stats_col1, stats_col2 = st.columns(2)

                with stats_col1:
                    st.metric("Cache Hit Rate", f"{stats.get('hit_rate', 0):.1%}")
                    st.metric("Total Embeddings", stats.get("total_embeddings", 0))

                with stats_col2:
                    st.metric("Cache Size", f"{stats.get('cache_size_mb', 0):.2f} MB")
                    st.metric("Daily Cost", f"${stats.get('daily_cost', 0):.6f}")

    with col3:
        if st.button("📈 Show Analytics"):
            st.balloons()

    # Display recent recommendations
    st.subheader("🎯 Today's Recommendations")

    try:
        # Get recommendations from database
        papers = asyncio.run(get_recent_recommendations())

        if papers:
            for i, paper in enumerate(papers[:10], 1):
                with st.container():
                    st.markdown(
                        f"""
                    <div class="paper-card">
                        <h4>#{i}. {paper.get('title', 'Unknown Title')}</h4>
                        <p><strong>Authors:</strong> {', '.join(paper.get('authors', [])[:3])}</p>
                        <p><strong>Category:</strong> 
                            <span class="category-tag">{paper.get('category', 'Unknown')}</span>
                        </p>
                        <p><strong>Score:</strong> {paper.get('score', 0):.3f}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Render full abstract with inline LaTeX support
                    render_latex_abstract(paper.get("abstract", ""), truncate=False)

                    # Rating interface with star buttons
                    current_rating = paper.get("rating", 0)
                    col_rate1, col_rate_margin, col_rate2 = st.columns([1, 2, 1])

                    with col_rate1:
                        star_rating_widget(paper["id"], current_rating)

                    with col_rate2:
                        pdf_url = paper.get("pdf_url", "")
                        if pdf_url:
                            st.link_button(
                                "📎 View PDF", pdf_url, use_container_width=True
                            )
                        else:
                            st.button(
                                "📎 No PDF", disabled=True, use_container_width=True
                            )

                    st.divider()
        else:
            st.info(
                "No recommendations available. Click 'Generate New Recommendations' to get started!"
            )

    except Exception as e:
        st.error(f"Error loading recommendations: {e}")


async def get_recent_recommendations():
    """Get recent recommendations from the database."""
    if not st.session_state.db_manager:
        return []

    try:
        # Use current scores snapshot from the papers table
        papers = await st.session_state.db_manager.get_recent_recommendations(limit=10)
        # Fallback to recent papers if no scored recommendations yet
        if not papers:
            papers = await st.session_state.db_manager.get_recent_papers(limit=10)
        return papers
    except AttributeError:
        # Older DB manager without method: fallback gracefully
        try:
            return await st.session_state.db_manager.get_recent_papers(limit=10)
        except Exception as e:
            st.error(f"Error fetching recent papers: {e}")
            return []
    except Exception as e:
        st.error(f"Error fetching recommendations: {e}")
        return []


def show_rating_interface():
    """Show interface for rating papers."""
    st.title("⭐ Rate Papers")
    st.markdown("Rate papers to improve your recommendations!")

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        category_filter = st.selectbox(
            "Filter by Category", ["All"] + config.arxiv_categories
        )

    with col2:
        rating_filter = st.selectbox(
            "Filter by Rating",
            ["All", "Unrated", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
        )

    # Get papers
    try:
        papers = asyncio.run(get_papers_for_rating(category_filter, rating_filter))

        if papers:
            st.info(f"Found {len(papers)} papers")

            for paper in papers:
                with st.expander(f"📄 {paper.get('title', 'Unknown Title')}"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"**Authors:** {', '.join(paper.get('authors', []))}"
                        )
                        st.markdown(f"**Category:** {paper.get('category', 'Unknown')}")
                        st.markdown(
                            f"**Published:** {paper.get('published_date', 'Unknown')}"
                        )
                        # Show current score if available
                        score_val = paper.get("current_score")
                        score_text = (
                            f"{score_val:.3f}"
                            if isinstance(score_val, (int, float))
                            and score_val is not None
                            else "N/A"
                        )
                        st.markdown(f"**Score:** {score_text}")

                        # Render full abstract with inline LaTeX support
                        render_latex_abstract(paper.get("abstract", ""), truncate=False)

                        # Notes
                        notes = st.text_area(
                            "Notes (optional)",
                            value=paper.get("notes", ""),
                            key=f"notes_{paper['id']}",
                        )

                    with col2:
                        current_rating = paper.get("rating", 0)

                        # Star rating with auto-save
                        star_rating_widget(
                            paper["id"], current_rating, notes if notes else None
                        )

                        # Direct links
                        if paper.get("arxiv_url"):
                            st.link_button(
                                "📖 ArXiv", paper["arxiv_url"], use_container_width=True
                            )
                        if paper.get("pdf_url"):
                            st.link_button(
                                "📎 PDF", paper["pdf_url"], use_container_width=True
                            )

        else:
            st.info("No papers found with the selected filters.")

    except Exception as e:
        st.error(f"Error loading papers: {e}")


async def get_papers_for_rating(category_filter: str, rating_filter: str):
    """Get papers for the rating interface with filters."""
    if not st.session_state.db_manager:
        return []

    try:
        # Get all papers (this would need filtering implemented in database.py)
        papers = await st.session_state.db_manager.get_all_papers()

        logger.info(f"Found {len(papers)} papers")
        logger.info(papers[0].get("current_score"))

        # Apply filters (basic implementation)
        filtered_papers = []
        for paper in papers:
            if category_filter != "All" and paper.get("category") != category_filter:
                continue

            paper_rating = paper.get("rating", 0)
            if rating_filter != "All":
                if rating_filter == "Unrated" and paper_rating > 0:
                    continue
                elif rating_filter.endswith("Star") or rating_filter.endswith("Stars"):
                    expected_rating = int(rating_filter.split()[0])
                    if paper_rating != expected_rating:
                        continue

            filtered_papers.append(paper)

        return filtered_papers

    except Exception as e:
        st.error(f"Error filtering papers: {e}")
        return []


def show_search_interface():
    """Show search interface for papers."""
    st.title("🔍 Search Papers")

    # Search inputs
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Search papers by title, abstract, or authors")

    with col2:
        if st.button("🔍 Search"):
            if search_query:
                try:
                    papers = asyncio.run(search_papers(search_query))
                    st.session_state.search_results = papers
                except Exception as e:
                    st.error(f"Search error: {e}")

    # Display search results
    if hasattr(st.session_state, "search_results") and st.session_state.search_results:
        st.subheader(f"Found {len(st.session_state.search_results)} papers")

        for paper in st.session_state.search_results:
            with st.container():
                st.markdown(f"**{paper.get('title', 'Unknown')}**")
                st.markdown(f"*{', '.join(paper.get('authors', [])[:3])}*")
                st.markdown(f"Category: {paper.get('category', 'Unknown')}")

                # Rating display
                current_rating = paper.get("rating", 0)
                if current_rating > 0:
                    st.markdown(
                        f"Your rating: {'⭐' * current_rating} ({current_rating}/5)"
                    )
                else:
                    st.markdown("⭐ Not rated yet")

                st.divider()


async def search_papers(query: str):
    """Search papers in the database."""
    if not st.session_state.db_manager:
        return []

    try:
        # This would need to be implemented in database.py
        # For now, get all papers and do basic filtering
        all_papers = await st.session_state.db_manager.get_all_papers()

        results = []
        query_lower = query.lower()

        for paper in all_papers:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            authors = " ".join(paper.get("authors", [])).lower()

            if (
                query_lower in title
                or query_lower in abstract
                or query_lower in authors
            ):
                results.append(paper)

        return results

    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def show_settings():
    """Show settings and configuration."""
    st.title("⚙️ Settings")

    # System Configuration
    st.subheader("System Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**OpenAI Configuration**")
        api_key_status = "✅ Set" if config.openai_api_key else "❌ Missing"
        st.markdown(f"API Key: {api_key_status}")
        st.markdown(f"Model: `{config.embedding_model}`")
        st.markdown(f"Budget Limit: ${config.openai_budget_limit:.2f}")

    with col2:
        st.markdown("**ArXiv Configuration**")
        st.markdown(f"Categories: {', '.join(config.arxiv_categories)}")
        st.markdown(f"Max Daily Papers: {config.max_daily_papers}")
        st.markdown(f"Database: `{config.database_path}`")

    # User Preferences
    st.subheader("User Preferences")

    # This would allow users to modify preferences
    st.info("User preference management will be implemented here.")

    # Cache Management
    st.subheader("Cache Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear Embedding Cache"):
            if st.session_state.embedding_manager:
                asyncio.run(st.session_state.embedding_manager.clear_cache())
                st.success("Cache cleared!")

    with col2:
        if st.button("Show Cache Stats"):
            if st.session_state.embedding_manager:
                stats = st.session_state.embedding_manager.get_cache_stats()
                st.json(stats)

    with col3:
        if st.button("Migrate Embeddings"):
            st.info("Embeddings are already using HDF5 format!")


def show_analytics():
    """Show analytics and statistics."""
    st.title("📊 Analytics")

    try:
        # Get analytics data
        analytics_data = asyncio.run(get_analytics_data())

        # Rating distribution
        if analytics_data.get("rating_distribution"):
            st.subheader("Rating Distribution")

            ratings_df = pd.DataFrame(analytics_data["rating_distribution"])
            fig = px.bar(
                ratings_df,
                x="rating",
                y="count",
                title="Distribution of Your Ratings",
                color="count",
                color_continuous_scale="viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Category preferences
        if analytics_data.get("category_preferences"):
            st.subheader("Category Preferences")

            cat_df = pd.DataFrame(analytics_data["category_preferences"])
            fig = px.pie(
                cat_df,
                values="avg_rating",
                names="category",
                title="Average Rating by Category",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Timeline
        if analytics_data.get("rating_timeline"):
            st.subheader("Rating Timeline")

            timeline_df = pd.DataFrame(analytics_data["rating_timeline"])
            timeline_df["date"] = pd.to_datetime(timeline_df["date"])

            fig = px.line(
                timeline_df, x="date", y="daily_ratings", title="Daily Rating Activity"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Ratings", analytics_data.get("total_ratings", 0))

        with col2:
            st.metric("Average Rating", f"{analytics_data.get('avg_rating', 0):.2f}")

        with col3:
            st.metric("Total Papers", analytics_data.get("total_papers", 0))

        with col4:
            st.metric(
                "Completion Rate", f"{analytics_data.get('completion_rate', 0):.1%}"
            )

    except Exception as e:
        st.error(f"Error loading analytics: {e}")


async def get_analytics_data():
    """Get analytics data from the database."""
    if not st.session_state.db_manager:
        return {}

    try:
        # This would need proper analytics methods in database.py
        # For now, return dummy data structure
        return {
            "rating_distribution": [{"rating": i, "count": 0} for i in range(1, 6)],
            "category_preferences": [],
            "rating_timeline": [],
            "total_ratings": 0,
            "avg_rating": 0.0,
            "total_papers": 0,
            "completion_rate": 0.0,
        }

    except Exception as e:
        st.error(f"Analytics error: {e}")
        return {}


if __name__ == "__main__":
    main()
