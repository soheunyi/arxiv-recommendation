# 📚 ArXiv Recommendation System - Web Interface

## Overview

The Streamlit web interface provides an intuitive way to interact with your ArXiv recommendation system. It offers paper rating functionality, recommendation viewing, analytics, and system management through a modern web interface.

## 🚀 Quick Start

### Method 1: Using the Launcher Script
```bash
python run_web.py
```

### Method 2: Direct Streamlit Command
```bash
python -m streamlit run src/web/app.py
```

### Method 3: Using Script Command (after installation)
```bash
uv run arxiv-web
```

The web interface will be available at: **http://localhost:8501**

## 📱 Features

### 🏠 Dashboard
- **Generate New Recommendations**: Run the recommendation workflow to get fresh paper suggestions
- **View Today's Recommendations**: Browse your personalized recommendations with ratings and metadata
- **Quick Actions**: Access cache stats, analytics, and system information
- **Paper Cards**: Rich display of papers with titles, authors, abstracts, categories, and scores

### ⭐ Rate Papers
- **Interactive Rating System**: 1-5 star rating scale for each paper
- **Filtering Options**: 
  - Filter by category (cs.AI, cs.LG, etc.)
  - Filter by rating status (All, Unrated, 1-5 Stars)
- **Add Notes**: Optional text notes for each rating
- **Bulk Rating**: Efficiently rate multiple papers
- **Direct Links**: Quick access to arXiv pages and PDFs

### 🔍 Search Papers
- **Full-Text Search**: Search through titles, abstracts, and authors
- **Real-Time Results**: Instant search results with highlighting
- **Rating Display**: See your existing ratings for search results
- **Quick Rating**: Rate papers directly from search results

### ⚙️ Settings
- **System Configuration**: View OpenAI API settings, model configuration, and budget limits
- **ArXiv Settings**: Check categories, paper limits, and database paths
- **Cache Management**: 
  - Clear embedding cache
  - View detailed cache statistics
  - Manage storage usage
- **User Preferences**: Configure personal settings (future feature)

### 📊 Analytics
- **Rating Distribution**: Visual breakdown of your rating patterns
- **Category Preferences**: See which research areas you prefer
- **Timeline Analysis**: Track your rating activity over time
- **Summary Metrics**: 
  - Total ratings given
  - Average rating score
  - Total papers in database
  - Rating completion rate

## 🎨 User Interface Features

### Modern Design
- **Responsive Layout**: Works on desktop and mobile devices
- **Clean Cards**: Organized paper information in readable cards
- **Interactive Elements**: Star ratings, buttons, and sliders
- **Category Tags**: Visual category labels with consistent styling
- **Color-Coded Metrics**: Easy-to-read statistics and status indicators

### Navigation
- **Sidebar Navigation**: Easy switching between different sections
- **Breadcrumb Trail**: Clear indication of current page
- **Quick Actions**: Common tasks accessible from multiple pages
- **Keyboard Shortcuts**: Streamlit's built-in navigation shortcuts

## 🔧 Technical Details

### Architecture
```
src/web/app.py                 # Main Streamlit application
├── Dashboard                  # Home page with recommendations
├── Rating Interface          # Paper rating functionality  
├── Search Interface          # Paper search and discovery
├── Settings                  # System configuration
└── Analytics                 # Usage statistics and insights
```

### Dependencies
- **Streamlit**: Web framework for Python data apps
- **Plotly**: Interactive charts and visualizations
- **Pandas**: Data manipulation for analytics
- **Asyncio**: Async integration with the database

### Database Integration
The web interface connects to your existing SQLite database and provides:
- Real-time access to papers and ratings
- Async database operations for better performance
- Transaction safety for rating updates
- Efficient querying with proper indexing

## 📊 Rating System

### How Ratings Work
1. **Scale**: 1-5 stars (1 = not interested, 5 = very interested)
2. **Impact**: Ratings ≥4 stars are used to build your preference profile
3. **Storage**: Ratings are stored with timestamps and optional notes
4. **Uniqueness**: One rating per paper (updates replace previous ratings)

### Rating Interface
- **Visual Stars**: Click on stars to set ratings
- **Slider Control**: Alternative slider input for precise rating
- **Notes Field**: Add personal comments about papers
- **Bulk Operations**: Rate multiple papers efficiently
- **Filter Options**: Focus on unrated or specific rating levels

### Recommendation Impact
Your ratings directly improve future recommendations:
- **Preference Learning**: High-rated papers (4-5 stars) train your preference model
- **Content-Based Filtering**: Similar papers to your highly-rated ones get boosted
- **Category Weighting**: Your preferred research areas get higher priority
- **Continuous Learning**: Each new rating immediately improves the model

## 🔍 Search and Discovery

### Search Functionality
- **Full-Text Search**: Searches titles, abstracts, and author names
- **Real-Time Results**: Instant search as you type
- **Relevance Ranking**: Results ordered by search relevance
- **Highlight Matching**: Key terms highlighted in results

### Filtering Options
- **By Category**: Filter to specific arXiv categories
- **By Rating**: Show only rated/unrated papers
- **By Date**: Filter by publication or addition date
- **Combined Filters**: Use multiple filters simultaneously

## 📈 Analytics Dashboard

### Available Metrics
- **Rating Distribution**: How you rate papers across the 1-5 scale
- **Category Preferences**: Which research areas you rate highly
- **Activity Timeline**: Your rating activity over time
- **Completion Metrics**: Percentage of papers you've rated

### Visualization Types
- **Bar Charts**: Rating distribution and category counts
- **Pie Charts**: Category preference breakdown
- **Line Charts**: Timeline of rating activity
- **Metric Cards**: Key performance indicators

## 🚀 Getting Started Guide

### First Time Setup
1. **Start the Web Interface**:
   ```bash
   python run_web.py
   ```

2. **Generate Recommendations**:
   - Click "Generate New Recommendations" on the dashboard
   - Wait for the system to fetch and process papers
   - View your personalized recommendations

3. **Start Rating Papers**:
   - Go to "Rate Papers" section
   - Browse through papers and add ratings
   - Use filters to focus on specific categories

4. **Explore Analytics**:
   - Visit the Analytics section after rating several papers
   - View your preference patterns and activity

### Daily Workflow
1. **Check New Recommendations**: Start with the dashboard to see fresh suggestions
2. **Rate Interesting Papers**: Rate papers that catch your attention
3. **Search for Specific Topics**: Use search to find papers on current interests
4. **Review Analytics**: Periodically check your rating patterns

## 🔧 Troubleshooting

### Common Issues

#### "Failed to initialize system components"
- **Cause**: Database or configuration issues
- **Solution**: 
  ```bash
  python scripts/setup_database.py
  ```

#### "No recommendations available"
- **Cause**: No papers in database or recommendation system not run
- **Solution**: Click "Generate New Recommendations" or run:
  ```bash
  python main.py run
  ```

#### "OpenAI API Key Missing"
- **Cause**: API key not configured
- **Solution**: Set environment variable:
  ```bash
  export OPENAI_API_KEY="your-key-here"
  ```

#### Web Interface Won't Start
- **Cause**: Missing dependencies
- **Solution**: Install web dependencies:
  ```bash
  uv pip install streamlit plotly altair
  ```

### Performance Tips
- **Cache Utilization**: The interface caches data for better performance
- **Batch Rating**: Rate multiple papers in one session for efficiency
- **Regular Cleanup**: Periodically clear old cache data in Settings

## 🔮 Future Enhancements

### Planned Features
- **User Profiles**: Multiple user support with separate preferences
- **Collaboration**: Share ratings and recommendations with colleagues
- **Export Features**: Export ratings and recommendations to various formats
- **Advanced Analytics**: More detailed insights and trend analysis
- **Custom Filters**: Save and reuse complex filter combinations
- **Batch Operations**: Bulk rating and management tools

### Integration Roadmap
- **Zotero Integration**: Direct import/export with Zotero libraries
- **RSS Feeds**: Custom RSS feeds based on your preferences
- **Email Notifications**: Daily/weekly recommendation emails
- **Mobile App**: Native mobile application
- **Browser Extension**: In-browser paper rating and saving

## 📚 API Reference

### Key Functions
- `show_dashboard()`: Main recommendation display
- `show_rating_interface()`: Paper rating functionality
- `show_search_interface()`: Search and discovery
- `show_analytics()`: Statistics and insights
- `update_paper_rating(paper_id, rating, notes)`: Rating updates

### Database Methods
- `get_papers_with_ratings()`: Fetch papers with rating data
- `get_recent_papers(limit)`: Get newest papers
- `search_papers(query)`: Full-text search
- `store_user_rating(paper_id, rating, notes)`: Save ratings

## 💡 Tips and Best Practices

### Effective Rating Strategy
1. **Be Consistent**: Use the same criteria for rating papers
2. **Rate Regularly**: Rate papers as you discover them
3. **Use the Full Scale**: Don't just use 3-5 stars, use 1-2 for uninteresting papers
4. **Add Notes**: Use notes for papers you want to remember
5. **Review Periodically**: Check your analytics to understand your preferences

### System Optimization
- **Regular Cache Cleanup**: Clear cache monthly for optimal performance
- **Database Maintenance**: Monitor database size and performance
- **Rating Coverage**: Aim for rating at least 20% of papers for good recommendations
- **Category Balance**: Rate papers across different categories for diverse recommendations

---

**🎉 Enjoy your personalized arXiv recommendation experience!**

For more information, see the main [README.md](README.md) and [API_REFERENCE.md](API_REFERENCE.md).