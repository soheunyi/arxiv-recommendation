# Intelligent ArXiv Paper Collection System

## Overview

This system provides intelligent, GPT-powered paper collection from arXiv for any research topic. It uses OpenAI's GPT-4o with structured outputs to generate sophisticated search queries, then executes them against the arXiv API to build comprehensive paper collections.

## Key Features

- **🤖 GPT-Powered Query Generation**: Automatically generates optimized search queries using GPT-4o
- **📊 Structured Outputs**: Uses JSON Schema to ensure reliable, validated responses
- **🎯 Intelligent Filtering**: Relevance filtering using GPT-suggested keywords
- **🔄 Deduplication**: Automatic removal of duplicate papers across queries
- **⚡ Rate Limiting**: Respectful API usage with proper rate limiting
- **💾 Configuration Save/Load**: Save and reuse query configurations
- **📈 Progress Tracking**: Real-time collection statistics and summaries

## Quick Start

### 1. Collect Papers on Any Topic

```bash
# Collect papers on a new topic (cleans database)
python scripts/collect_papers_by_topic.py "federated learning" --clean-db

# Preview what would be collected without executing
python scripts/collect_papers_by_topic.py "transformer models" --preview

# Collect with custom limits
python scripts/collect_papers_by_topic.py "graph neural networks" --max-papers 10 --max-queries 8
```

### 2. Generate and Preview Queries

```bash
# Generate and preview queries for any topic
python scripts/gpt_query_generator.py "quantum machine learning" --preview

# Save query configuration for reuse
python scripts/gpt_query_generator.py "reinforcement learning" --save configs/rl_queries.json
```

### 3. Use Saved Configurations

```bash
# Use a previously saved configuration
python scripts/collect_papers_by_topic.py "reinforcement learning" --config configs/rl_queries.json

# Append to existing database (don't clean)
python scripts/collect_papers_by_topic.py "computer vision" --no-clean
```

## Command Reference

### collect_papers_by_topic.py

Main script for collecting papers on any research topic.

**Basic Usage:**
```bash
python scripts/collect_papers_by_topic.py TOPIC [options]
```

**Key Options:**
- `--clean-db`: Clean database before collection (default)
- `--no-clean`: Append to existing database
- `--preview`: Show collection plan without executing
- `--dry-run`: Collect papers but don't store in database
- `--max-papers N`: Maximum papers per query (default: 20)
- `--max-queries N`: Maximum queries to generate (default: 15)
- `--config FILE`: Load queries from configuration file
- `--save-config FILE`: Save generated queries to file
- `--save-report FILE`: Save detailed collection report
- `--skip-filtering`: Skip relevance filtering

### gpt_query_generator.py

Standalone query generator for testing and configuration creation.

**Basic Usage:**
```bash
python scripts/gpt_query_generator.py TOPIC [options]
```

**Key Options:**
- `--preview`: Preview generated queries
- `--save FILE`: Save queries to configuration file
- `--max-queries N`: Maximum number of queries to generate

### backup_database.py

Database backup and restoration utility.

**Usage:**
```bash
# Create backup
python scripts/backup_database.py backup

# List available backups  
python scripts/backup_database.py list

# Restore ratings from backup
python scripts/backup_database.py restore --backup-file data/user_ratings_backup_20240813_123456.json
```

## Example Workflows

### Research a New Field

```bash
# 1. Preview what we'll collect
python scripts/collect_papers_by_topic.py "causal inference" --preview

# 2. Collect papers and save configuration
python scripts/collect_papers_by_topic.py "causal inference" --clean-db --save-config configs/causal_inference.json

# 3. Generate collection report
python scripts/collect_papers_by_topic.py "causal inference" --config configs/causal_inference.json --save-report reports/causal_inference_report.json
```

### Compare Multiple Topics

```bash
# Collect different topics and compare
python scripts/collect_papers_by_topic.py "optimal transport" --clean-db --save-report reports/ot_report.json
python scripts/collect_papers_by_topic.py "generative models" --clean-db --save-report reports/gen_models_report.json
python scripts/collect_papers_by_topic.py "graph neural networks" --clean-db --save-report reports/gnn_report.json
```

### Incremental Collection

```bash
# Start with core topic
python scripts/collect_papers_by_topic.py "machine learning" --clean-db

# Add related topics without cleaning
python scripts/collect_papers_by_topic.py "deep learning" --no-clean --max-queries 8
python scripts/collect_papers_by_topic.py "neural networks" --no-clean --max-queries 6
```

## Generated Query Examples

### For "Optimal Transport":
- `ti:"optimal transport"` - Papers with exact phrase in title
- `all:"Wasserstein distance"` - Wasserstein distance mentions
- `cat:math.OC AND all:"optimal transport"` - OT papers in optimization category
- `au:"Villani" AND all:"optimal transport"` - Papers by key researcher
- `all:"Sinkhorn algorithm"` - Computational methods

### For "Quantum Computing":
- `ti:"quantum computing"` - Core topic in titles
- `abs:"quantum computing" AND cat:quant-ph` - QC in quantum physics
- `ti:"quantum algorithms" OR abs:"quantum circuits"` - Related concepts
- `au:"John Preskill" AND all:"quantum"` - Expert contributions
- `ti:"quantum machine learning"` - Intersection fields

## Configuration Files

Query configurations are saved as JSON files with this structure:

```json
{
  "topic": "machine learning",
  "search_queries": [
    {
      "query": "ti:\"machine learning\"",
      "priority": "high",
      "description": "Core ML papers by title"
    }
  ],
  "categories": ["cs.LG", "stat.ML"],
  "filter_keywords": ["machine learning", "ML", "neural"],
  "related_terms": ["deep learning", "neural networks"]
}
```

## Best Practices

### Topic Selection
- Use specific, well-defined research topics
- Avoid overly broad terms like "AI" or "computer science"
- Include methodology terms when relevant (e.g., "MCMC methods", "transformer models")

### Collection Strategy
- Start with `--preview` to understand what will be collected
- Use `--max-papers 10-15` for focused, high-quality collections
- Save configurations for topics you'll revisit
- Use `--no-clean` to build diverse, multi-topic databases

### Quality Control
- Review sample papers in collection summaries
- Use relevance filtering (enabled by default)
- Check category distributions to ensure topic coverage
- Save reports for analysis and comparison

## Troubleshooting

### Common Issues

**GPT API Errors:**
- Ensure `OPENAI_API_KEY` is set in environment or `.env` file
- Check API quota and billing status
- Verify internet connectivity

**Empty Results:**
- Try broader or more common terminology
- Check if topic exists in arXiv
- Use `--skip-filtering` to see all results

**Database Issues:**
- Use `backup_database.py` before major operations
- Check file permissions on data directory
- Restart API server after database changes

### Performance Optimization

- Use `--max-papers 15` for faster collection
- Reduce `--max-queries` for narrower focus
- Use saved configs to avoid repeated GPT calls
- Run collections during off-peak hours

## System Architecture

```
User Input (Topic)
    ↓
GPT Query Generator (gpt_query_generator.py)
    ↓ 
Structured JSON Schema Validation
    ↓
ArXiv API Client (arxiv_client.py)
    ↓
Relevance Filtering
    ↓
Deduplication
    ↓
Database Storage (database.py)
    ↓
Frontend Display
```

The system is designed to be modular, allowing each component to be used independently or as part of the complete pipeline.