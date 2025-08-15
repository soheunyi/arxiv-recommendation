# 🤖 OpenAI-Gemini Collaboration System

Advanced collaboration system that intelligently orchestrates OpenAI and Gemini for optimal query generation with cost efficiency and quality assurance.

## 🎯 Overview

The collaboration system provides four strategic approaches to combine the strengths of both LLM providers:

- **Cost Optimized**: Use Gemini primarily (33x cheaper), OpenAI for quality validation
- **Quality First**: Use OpenAI primarily, Gemini as backup
- **Parallel Compare**: Run both providers, select best result
- **Adaptive**: Automatically select optimal strategy based on context

## 💰 Cost Benefits

- **Gemini vs OpenAI**: 33x cheaper for both input and output tokens
- **Smart Validation**: Only use expensive OpenAI when quality threshold not met
- **Budget Tracking**: Real-time cost monitoring and budget management
- **Estimated Savings**: 70-90% cost reduction with maintained quality

## 🚀 Quick Start

### 1. API Usage

```bash
# Generate queries with collaboration
curl -X POST "http://localhost:8000/api/collaboration/generate-queries" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "large language models",
    "max_queries": 15,
    "strategy": "adaptive",
    "quality_threshold": 0.8,
    "date_from": "2024-01-01",
    "date_to": "2024-12-31"
  }'

# Get available strategies
curl "http://localhost:8000/api/collaboration/strategies"

# Check usage statistics
curl "http://localhost:8000/api/collaboration/usage-stats"

# Switch default provider
curl -X POST "http://localhost:8000/api/collaboration/switch-provider" \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini"}'
```

### 2. Python Integration

```python
from services.collaborative_service import collaborative_service, CollaborationStrategy

# Generate queries with collaboration
result = await collaborative_service.generate_queries_collaborative(
    topic="quantum computing",
    max_queries=15,
    strategy=CollaborationStrategy.ADAPTIVE,
    quality_threshold=0.8
)

print(f"Provider: {result.primary_provider}")
print(f"Quality: {result.quality_score}")
print(f"Cost: ${result.cost_estimate}")
print(f"Queries: {len(result.primary_result['search_queries'])}")
```

### 3. Test Script

```bash
# Run comprehensive collaboration tests
cd scripts
python test_collaboration.py
```

## 📊 Collaboration Strategies

### 🎯 Adaptive Strategy (Recommended)

Automatically selects the best approach based on:
- **Topic Complexity**: Complex topics → Parallel Compare
- **Budget Usage**: High usage → Cost Optimized  
- **Query Count**: Large requests → Quality First
- **Default**: Cost Optimized for most cases

**Selection Logic**:
```python
if budget_used > 80%:
    return COST_OPTIMIZED
elif complexity_score >= 3 or max_queries > 20:
    return PARALLEL_COMPARE
elif complexity_score >= 1:
    return QUALITY_FIRST
else:
    return COST_OPTIMIZED
```

### 💰 Cost Optimized Strategy

1. **Primary**: Generate queries with Gemini (cheap)
2. **Quality Check**: Evaluate result quality score
3. **Validation**: If quality < threshold, use OpenAI for validation
4. **Merge**: Combine results for optimal coverage

**Best For**: Budget-conscious scenarios, routine queries, high-volume usage

### 🏆 Quality First Strategy

1. **Primary**: Use OpenAI for highest quality
2. **Fallback**: If OpenAI fails, use Gemini backup
3. **Reliability**: Prioritizes proven performance

**Best For**: Critical applications, complex research topics, production use

### ⚡ Parallel Compare Strategy

1. **Parallel Execution**: Run both providers simultaneously
2. **Quality Comparison**: Evaluate both results
3. **Best Selection**: Choose highest quality result
4. **Merge Enhancement**: Combine unique queries from both

**Best For**: Complex topics, comprehensive coverage, when cost is not primary concern

## 🔧 Configuration

### Environment Variables

```bash
# Provider selection
LLM_PROVIDER=gemini  # or "openai"

# API Keys (both required for collaboration)
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

# Budget management
OPENAI_BUDGET_LIMIT=20.0

# Quality settings (optional)
COLLABORATION_QUALITY_THRESHOLD=0.8
COLLABORATION_DEFAULT_STRATEGY=adaptive
```

### Provider Configuration

```python
# config.py settings
openai_query_model = "gpt-4o"               # High quality
gemini_query_model = "gemini-2.5-flash"    # Cost effective

# Cost per 1K tokens
openai_costs = {"input": 2.5, "output": 10.0}
gemini_costs = {"input": 0.075, "output": 0.3}  # 33x cheaper!
```

## 📈 Quality Evaluation

The system evaluates query quality based on:

### Quality Metrics (0.0 - 1.0 scale)

1. **Query Count Score (30%)**: More queries = better coverage
2. **Query Diversity Score (30%)**: Different query types (title, abstract, author, category)
3. **Priority Distribution Score (20%)**: Good mix of high/medium/low priorities
4. **Metadata Completeness Score (20%)**: Categories, keywords, related terms

### Quality Thresholds

- **0.8+**: Excellent quality, no validation needed
- **0.6-0.8**: Good quality, optional validation
- **0.4-0.6**: Moderate quality, validation recommended
- **<0.4**: Poor quality, validation required

## 💡 Usage Patterns

### Research Phase
```python
# Comprehensive topic exploration
strategy = CollaborationStrategy.PARALLEL_COMPARE
quality_threshold = 0.9
```

### Development Phase
```python
# Balanced cost and quality
strategy = CollaborationStrategy.ADAPTIVE
quality_threshold = 0.7
```

### Production Phase
```python
# Cost optimization with quality assurance
strategy = CollaborationStrategy.COST_OPTIMIZED
quality_threshold = 0.8
```

## 📊 Monitoring & Analytics

### Usage Statistics

```json
{
  "providers": {
    "openai": {"requests": 10, "total_cost": 0.25},
    "gemini": {"requests": 40, "total_cost": 0.08}
  },
  "total_requests": 50,
  "total_cost": 0.33,
  "budget_used_percentage": 1.65,
  "cost_savings": {
    "absolute_savings": 2.12,
    "percentage_savings": 86.5
  }
}
```

### Performance Metrics

- **Execution Time**: Average response time per strategy
- **Quality Scores**: Quality distribution across providers
- **Cost Efficiency**: Cost per query by strategy
- **Success Rates**: Provider reliability metrics

## 🔍 Advanced Features

### Date Filtering Support

Both providers support arXiv date filtering:
```python
result = await collaborative_service.generate_queries_collaborative(
    topic="transformers",
    date_from="2024-01-01",  # YYYY-MM-DD format
    date_to="2024-12-31",
    strategy=CollaborationStrategy.ADAPTIVE
)
```

### Quality Validation

Automatic quality validation with configurable thresholds:
```python
# If Gemini quality < 0.8, validate with OpenAI
result = await collaborative_service.generate_queries_collaborative(
    topic="complex_topic",
    strategy=CollaborationStrategy.COST_OPTIMIZED,
    quality_threshold=0.8  # Trigger OpenAI validation
)
```

### Result Merging

Intelligent merging of results from multiple providers:
- Deduplicate similar queries
- Combine unique queries from both providers  
- Merge categories, keywords, and related terms
- Maintain quality while maximizing coverage

## 🚀 Best Practices

### Cost Optimization
1. Use **Adaptive** strategy for general purposes
2. Set appropriate quality thresholds (0.7-0.8)
3. Monitor budget usage regularly
4. Use **Cost Optimized** for high-volume operations

### Quality Assurance
1. Use **Quality First** for critical applications
2. Set higher thresholds (0.8+) for important queries
3. Use **Parallel Compare** for complex topics
4. Review quality scores and adjust thresholds

### Performance Tuning
1. Cache frequently used results
2. Use parallel execution for independent queries
3. Monitor execution times
4. Adjust strategies based on usage patterns

## 🔧 Troubleshooting

### Common Issues

**High Costs**
- Switch to Cost Optimized strategy
- Lower quality thresholds
- Use Gemini as default provider

**Low Quality Results**
- Increase quality thresholds
- Use Quality First strategy
- Enable parallel comparison

**Slow Performance**
- Check network connectivity
- Verify API key configuration
- Monitor rate limiting

### Debugging

```python
# Enable detailed logging
import logging
logging.getLogger("services.collaborative_service").setLevel(logging.DEBUG)

# Check provider status
from services.provider_factory import ProviderFactory
comparison = ProviderFactory.compare_providers()
print(comparison)
```

## 📚 API Reference

### Endpoints

- `POST /api/collaboration/generate-queries` - Generate collaborative queries
- `GET /api/collaboration/strategies` - List available strategies
- `GET /api/collaboration/usage-stats` - Get usage statistics
- `POST /api/collaboration/switch-provider` - Switch default provider

### Python Classes

- `CollaborativeService` - Main orchestration service
- `CollaborationStrategy` - Strategy enumeration
- `CollaborationResult` - Result data class
- `ProviderFactory` - Provider management

## 🎉 Next Steps

1. **Test the System**: Run `python scripts/test_collaboration.py`
2. **Configure Strategies**: Set up your preferred collaboration approach
3. **Monitor Usage**: Track costs and quality metrics
4. **Optimize Settings**: Adjust thresholds based on your needs
5. **Scale Up**: Use for production query generation

The collaboration system is ready to help you get the best results while keeping costs under control! 🚀