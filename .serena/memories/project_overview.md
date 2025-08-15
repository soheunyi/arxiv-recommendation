# ArXiv Recommendation System - Project Overview

## Purpose
Personal arXiv recommendation system that automatically collects, scores, and recommends research papers based on user preferences. Uses LLM embeddings for semantic understanding and multi-agent workflow for intelligent paper discovery.

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI for REST API
- **Database**: SQLite with aiosqlite for async operations
- **LLM Integration**: OpenAI GPT-4o and Google Gemini 2.5 Flash
- **Embeddings**: OpenAI text-embedding-3-small
- **Multi-Agent**: Autogen framework
- **Package Manager**: UV (modern Python package manager)

### Frontend  
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **State Management**: Redux Toolkit with RTK Query
- **Styling**: Tailwind CSS
- **UI Components**: Headless UI, Heroicons
- **Animation**: Framer Motion

## Key Features
- Automatic daily paper collection from arXiv
- Personalized recommendations using MMR algorithm
- User rating system for preference learning
- Cost-optimized LLM provider selection
- Rich CLI interface with cost analysis
- Web interface for paper browsing and management

## Project Status
- Phase 1 (Core Pipeline): Complete ✅
- Phase 2 (User Interface): In Progress 🔄
- Phase 3 (Production Features): Planned 📈

## Architecture
- Multi-agent system: DataAgent → RecommendationAgent → Coordinator
- Async operations throughout for performance
- Local embedding cache for cost optimization
- Rate-limited API calls respecting arXiv limits