# ArXiv Recommendation System

Personal arXiv recommendation system using OpenAI embeddings and multi-agent workflow. **Status: Core Implementation Complete** ✅

## 🚀 Implementation Status

### ✅ **Phase 1 Complete** - Core Pipeline (Week 1)
- **ArXiv Client**: Async API wrapper with rate limiting and 50+ category support
- **Database**: SQLite with 6 tables, async operations, user preferences, and statistics  
- **Embeddings**: OpenAI integration with 30-day local caching and batch processing
- **Recommendations**: Multi-factor scoring with MMR diversity optimization
- **Multi-Agent System**: Complete DataAgent → RecommendationAgent → Coordinator workflow
- **CLI Interface**: Rich terminal UI with config management and cost analysis
- **Testing**: Full integration testing and database setup automation

### 🔄 **Phase 2 Next** - User Interface (Week 3) 
- **Streamlit Web App**: Visual paper browsing, rating system, preference management
- **Enhanced Personalization**: Improved user preference learning and recommendation tuning
- **Daily Automation**: Cron job setup and automated monitoring

### 📈 **Phase 3 Future** - Production Features (Week 4)
- **Testing Suite**: Unit and integration tests with coverage reporting  
- **Deployment**: Local deployment guide and monitoring setup
- **Advanced Features**: Email notifications, export options, mobile interface

## ✨ Features

- 🤖 **Multi-Agent Architecture**: DataAgent, RecommendationAgent, and Coordinator working together
- 🧠 **Dual LLM Support**: OpenAI GPT-4o and Google Gemini 2.5 Flash with provider selection
- 💰 **Cost-Optimized**: Gemini 33x cheaper than OpenAI for query generation (~$0.01-0.60/month)
- 🎯 **Personalized Recommendations**: MMR algorithm with similarity, novelty, diversity, and quality scoring
- 📊 **Rich CLI Interface**: Beautiful terminal output with tables, progress, and cost analysis
- ⚡ **Async Performance**: Rate-limited API calls with local embedding cache
- 🛡️ **Production Ready**: Comprehensive error handling, logging, and budget controls

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Gemini API key (or both)
- UV package manager

### Installation

1. **Clone and setup the project**:
```bash
git clone <your-repo-url>
cd arxiv-recommendation
```

2. **Install UV** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Initialize the project**:
```bash
# Install dependencies
uv sync

# Install development dependencies (optional)
uv sync --group dev

# Install web interface dependencies
uv sync --group web
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys:
# - Set LLM_PROVIDER to "openai" or "gemini"  
# - Add OPENAI_API_KEY or GEMINI_API_KEY (or both)
```

### Usage

#### Command Line Interface

```bash
# Setup database (first time only)
uv run python scripts/setup_database.py

# Run daily paper collection and recommendations
uv run python main.py run

# Show current configuration
uv run python main.py config-info

# Analyze costs and budget
uv run python main.py cost

# Test core pipeline
uv run python scripts/test_pipeline.py
```

#### Web Interface

```bash
# Launch both frontend and backend servers with intelligent port detection
uv run python start_servers.py

# Automatically kill existing servers without prompting
uv run python start_servers.py --force

# Debug mode with verbose logging
uv run python start_servers.py --debug

# Debug frontend startup issues only
uv run python start_servers.py --debug-frontend

# Note: The convenience script 'uv run arxiv-serve' may have path resolution issues
# Use the direct python command above for best results

# Test server startup without running
uv run python test_startup.py

# Alternatively, run servers separately:
# Backend API server
uv run uvicorn backend.api:app --reload --port 8000

# Frontend dev server (in separate terminal)
cd frontend && npm run dev
```

#### Python API

```python
import asyncio
from backend.arxiv_recommendation import run_recommendation_system

# Run the complete workflow
async def main():
    results = await run_recommendation_system()
    print(f"Generated {len(results.get('recommendations', []))} recommendations")

asyncio.run(main())
```

## Architecture

### Multi-Agent System

- **DataAgent**: Collects papers from arXiv API, handles preprocessing
- **RecommendationAgent**: Generates embeddings and recommendations  
- **Coordinator**: Orchestrates workflow between agents

### Cost Optimization

- Uses `text-embedding-3-small` ($0.00002/1K tokens)
- **Actual costs**: $0.01-0.60/month with intelligent caching
- **30-day embedding cache** with 80%+ hit rates
- **Budget controls**: Configurable daily/monthly limits with warnings
- **Batch processing**: Optimized API usage with rate limiting

### Data Flow

1. **Collection**: DataAgent fetches new papers from arXiv
2. **Embedding**: RecommendationAgent generates semantic embeddings
3. **User Preferences**: Computes user preference embeddings (alltime vs recent)
4. **Matching**: Similarity search against user preferences
5. **Ranking**: Personalized ranking based on ratings history
6. **Delivery**: Web interface or CLI output

### User Preference Embeddings

The system computes user preference embeddings using **two complementary approaches**:

#### **All-time Embeddings**
- Uses **all rated papers** with equal weighting
- Provides stable, long-term preference representation
- Requires minimum 5 rated papers to generate
- Good for discovering papers in established interest areas

#### **Recent Embeddings** 
- Focuses on **recent ratings** (last 30 days by default)
- Uses exponential time decay weighting (newer = higher weight)
- Captures evolving interests and research direction changes  
- Requires minimum 3 recent ratings to generate

#### **Adaptive Mode (Default)**
- **Combines multiple time windows** with EMA-style weighting:
  - Recent (7 days): 70% weight - captures current interests
  - Medium (30 days): 20% weight - balances recent trends
  - Long-term (365 days): 10% weight - maintains stable preferences
- Automatically adapts to available rating history
- Provides best balance between stability and responsiveness
- Falls back gracefully when insufficient data in any window

#### **Usage Examples**
```python
from backend.arxiv_recommendation.preferences import PreferenceManager, PreferenceMode

preference_manager = PreferenceManager()

# Get all-time preference embedding
alltime_pref = await preference_manager.get_preference_embedding(PreferenceMode.ALL_TIME)

# Get recent preference embedding  
recent_pref = await preference_manager.get_preference_embedding(PreferenceMode.RECENT)

# Get adaptive preference embedding (recommended)
adaptive_pref = await preference_manager.get_preference_embedding(PreferenceMode.ADAPTIVE)
```

The preference embeddings are automatically updated by the scheduler and cached for 6 hours for optimal performance.

## Configuration

Edit `.env` file:

```env
# Required
OPENAI_API_KEY=your_key_here

# Optional customization
ARXIV_CATEGORIES=cs.AI,cs.LG,cs.CL  # Your interests
MAX_DAILY_PAPERS=100               # Papers per day
OPENAI_BUDGET_LIMIT=20.0          # Monthly budget
```

## Development

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code  
uv run ruff check .

# Type checking
uv run mypy src/

# Run tests
uv run pytest
```

### Adding Dependencies

```bash
# Production dependency
uv add requests

# Development dependency  
uv add --group dev pytest

# Optional dependency group
uv add --group performance faiss-gpu
```

## Project Structure

```
arxiv-recommendation/
├── backend/arxiv_recommendation/   # 🔧 Python backend (ML/API)
│   ├── agents.py                  # ✅ Multi-agent system (DataAgent, RecommendationAgent, Coordinator)
│   ├── arxiv_client.py            # ✅ Async arXiv API client with rate limiting
│   ├── embeddings.py              # ✅ OpenAI embedding manager with caching
│   ├── recommendations.py         # ✅ MMR-based recommendation engine
│   ├── database.py                # ✅ Async SQLite operations with full schema
│   └── config.py                  # ✅ Environment-based configuration
├── frontend/                      # ⚛️ React TypeScript frontend
│   ├── src/components/            # UI components (papers, rating, analytics)
│   ├── src/pages/                 # Route-level pages
│   ├── src/services/              # API communication layer
│   ├── src/store/                 # Redux state management
│   └── src/types/                 # TypeScript definitions
├── scripts/                       # 🛠️ Utility scripts
│   ├── setup_database.py          # ✅ Database initialization
│   └── test_pipeline.py           # ✅ Core integration testing
├── data/                          # 📊 Local data storage (auto-created)
│   ├── papers.db                  # SQLite database
│   └── embeddings/                # Embedding cache
├── main.py                        # 🖥️ CLI interface with Rich output
├── start_servers.py               # 🚀 Development server manager
├── pyproject.toml                 # 📦 UV-based modern Python config
└── .env.example                   # ✅ Environment template
```

## Cost Monitoring

The system includes built-in cost tracking:

- Daily cost estimates logged
- Budget warnings when approaching limits
- API usage statistics
- Optimization suggestions

Monitor costs with:
```bash
# View current cost analysis
uv run python main.py cost

# Check configuration and API key status
uv run python main.py config-info
```

## 📚 Documentation

### Quick References
- **[Implementation Plan](IMPLEMENTATION_PLAN.md)** - Current status and roadmap
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all components
- **[Environment Setup](.env.example)** - Configuration template

### Component Documentation
- **ArXiv Client**: Async API wrapper with rate limiting and category validation
- **Database Manager**: SQLite operations with user preferences and statistics  
- **Embedding Manager**: OpenAI integration with 30-day caching and cost tracking
- **Recommendation Engine**: Multi-factor scoring with MMR diversity optimization
- **Multi-Agent System**: Complete workflow orchestration with Rich logging

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`uv run python scripts/test_pipeline.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## Acknowledgments

This project utilizes the [arXiv.org](https://arxiv.org/) e-print service for accessing academic papers. arXiv is owned and operated by Cornell University and is supported by the Simons Foundation and by the member institutions.

We are grateful to arXiv for providing free and open access to academic research, enabling projects like this to help researchers discover relevant papers and advance scientific knowledge.

## License

MIT License - see LICENSE file for details.