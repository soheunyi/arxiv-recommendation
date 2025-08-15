# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ Critical Rules

**NEVER modify test code without explicit permission**
**NEVER change API names and parameters**
**NEVER migrate data arbitrarily**

## Serena Activation
serena - activate_project when starting a new claude session.

## Essential Commands

### Python Package Management
- **Always use `uv` for Python execution and dependency management**
- Install dependencies: `uv sync` (production), `uv sync --group dev` (development)
- Run Python scripts: `uv run python <script>` or `uv run <command>`
- Add dependencies: `uv add <package>` (production), `uv add --group dev <package>` (dev)

### Development Workflow
```bash
# Setup (first time)
uv sync --group dev
uv run python scripts/setup_database.py

# Core development
uv run python main.py run                    # Run recommendation system
uv run python start_servers.py              # Start full stack (frontend + backend)
uv run python start_servers.py --force      # Force kill existing servers

# Code quality
uv run ruff format .                         # Format code
uv run ruff check .                          # Lint code
uv run mypy backend/src/                     # Type checking

# Testing (comprehensive educational suite)
uv run pytest tests/ -v                     # All tests with verbose output
uv run pytest tests/ -v -s                  # Include educational print statements
uv run pytest tests/unit/ -v                # Unit tests only
uv run pytest tests/integration/ -v         # Integration tests only
uv run pytest <path/to/test_file.py> -v     # Single test file
uv run pytest tests/ --cov=backend --cov-report=html  # Coverage report
```

### Database Operations
```bash
uv run python scripts/setup_database.py     # Initialize database
uv run python scripts/backup_database.py    # Backup database
uv run python scripts/migrate_embeddings.py # Migrate embedding cache
```

## High-Level Architecture

### Core System Design
This is a **full-stack ArXiv recommendation system** with Python backend and React TypeScript frontend, featuring multi-LLM collaboration (OpenAI GPT-4o + Google Gemini 2.5 Flash).

### Backend Architecture (`backend/src/`)
- **Multi-Agent System**: `agents.py` orchestrates DataAgent → RecommendationAgent → Coordinator workflow
- **LLM Integration**: Dual provider support with cost optimization (Gemini 33x cheaper than OpenAI)
  - `services/collaborative_service.py`: OpenAI-Gemini orchestration with 4 strategies
  - `services/query_service.py`: OpenAI query generation
  - `services/gemini_query_service.py`: Gemini query generation
  - `services/provider_factory.py`: Provider abstraction layer
- **Data Pipeline**: ArXiv API → Embedding generation → Similarity search → Personalized ranking
  - `arxiv_client.py`: Async ArXiv API with rate limiting
  - `embeddings.py`: OpenAI embeddings with 30-day caching
  - `recommendations.py`: MMR algorithm with novelty/diversity scoring
- **Reference Tracking**: Citation network analysis for paper relationships
  - `services/reference_service.py`: HTML parsing for arXiv citations
  - Database schema supports both ArXiv and external references
- **Storage**: `database.py` with async SQLite operations, comprehensive schema for papers/ratings/references
- **Personalization**: `preferences.py` with adaptive preference embeddings (all-time, recent, adaptive modes)

### Frontend Architecture (`frontend/`)
- **React TypeScript** with modern patterns (hooks, context, async/await)
- **Redux Toolkit** for state management (papers, ratings, collections)
- **UI Components**: Paper cards, rating system, reference tracking, collaboration settings
- **API Layer**: Services communicate with FastAPI backend

### Key Data Flows
1. **Paper Collection**: User query → LLM query generation → ArXiv API → Reference extraction → Database storage
2. **Recommendations**: User preferences → Embedding similarity → MMR ranking → Personalized results
3. **Reference Tracking**: ArXiv HTML → Citation parsing → Network construction → Related paper discovery
4. **Collaboration**: Query refinement through OpenAI/Gemini strategies with cost optimization

### Cost Optimization Strategy
- **Primary**: Gemini for query generation (33x cheaper: ~$0.01-0.60/month vs $30/month OpenAI)
- **Validation**: OpenAI for quality assurance when Gemini quality < threshold
- **Caching**: 30-day embedding cache with 80%+ hit rates
- **Budget Controls**: Configurable limits with usage tracking

## Testing Philosophy

### Educational Test Requirements
**Always write comprehensive tests for new implementations that serve both validation and educational purposes:**

1. **Unit Tests** (`tests/unit/`): Test individual components with detailed explanations
2. **Integration Tests** (`tests/integration/`): Demonstrate system interactions with real scenarios
3. **Educational Comments**: Each test includes "EDUCATIONAL:" sections explaining concepts
4. **Edge Case Coverage**: Handle malformed data, network failures, invalid inputs
5. **Performance Validation**: Include timing and efficiency measurements
6. **Realistic Test Data**: Use examples that mirror actual system usage

### Test Structure Pattern
```python
def test_feature_explanation(self, educational_print):
    """
    EDUCATIONAL: This test demonstrates [feature concept]
    
    Key learning points:
    - How [component] processes data
    - What transformations occur
    - How errors are handled gracefully
    """
    educational_print("Feature Name", "Detailed explanation...")
    # Test implementation with verbose logging
```

### Test Categories
- **Database**: Storage patterns, async operations, citation relationships
- **LLM Integration**: Provider coordination, cost optimization, quality validation
- **Reference System**: HTML parsing, citation networks, recommendation generation
- **API Integration**: Request/response cycles, error handling, data validation

## Configuration Management

### Environment Setup
- Copy `.env.example` to `.env`
- Configure LLM providers: `LLM_PROVIDER="gemini"` (recommended for cost) or `"openai"`
- Set API keys: `OPENAI_API_KEY` and/or `GEMINI_API_KEY`
- Customize categories: `ARXIV_CATEGORIES="cs.AI,cs.LG,cs.CL"`

### Development vs Production
- Development: Use `uv sync --group dev` for testing and debugging tools
- Web interface: `uv sync --group web` for Streamlit components
- Performance: `uv sync --group performance` for FAISS and Redis optimizations

## Common Development Patterns

### Async/Await Usage
All database operations and API calls use async patterns. Always use `await` for:
- Database operations (`db.store_papers()`, `db.get_papers()`)
- API calls (`arxiv_client.search_papers()`, LLM services)
- File I/O operations with embeddings and caching

### Error Handling
- Services implement graceful fallbacks (Gemini → OpenAI, cache miss → API call)
- Database operations use transactions for consistency
- Network operations include retry logic with exponential backoff

### Multi-LLM Coordination
- Use `CollaborativeService` for provider orchestration
- Strategies: Cost Optimized (default), Quality First, Parallel Compare, Adaptive
- Quality validation ensures output meets standards regardless of provider

This system prioritizes cost efficiency, educational value through comprehensive testing, and maintainable architecture through clear separation of concerns.
