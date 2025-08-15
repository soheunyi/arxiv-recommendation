# Gemini API Integration

This document describes the Google Gemini API integration for the ArXiv Recommendation System.

## Overview

The system now supports both OpenAI and Google Gemini as LLM providers for:
- **Query Generation**: Creating intelligent arXiv search queries
- **Embeddings**: Generating semantic embeddings for papers

## Quick Setup

### 1. Install Dependencies

```bash
# Install the new Google GenAI SDK
pip install google-genai
```

### 2. Get API Keys

- **Gemini**: Get a free API key from [Google AI Studio](https://ai.google.dev/)
- **OpenAI**: Get an API key from [OpenAI Platform](https://platform.openai.com/)

### 3. Configure Environment

Create a `.env` file with your API keys:

```bash
# Choose your provider
LLM_PROVIDER=gemini  # or "openai"

# Gemini API key (recommended for cost savings)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API key (if you prefer OpenAI)
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### CLI Scripts

The query generator now supports both providers:

```bash
# Use Gemini (default if configured)
python scripts/gpt_query_generator.py "quantum computing" --provider gemini

# Use OpenAI
python scripts/gpt_query_generator.py "quantum computing" --provider openai

# Compare providers
python scripts/gpt_query_generator.py --compare-providers

# Test integration
python scripts/test_gemini_integration.py --compare
```

### Python Code

```python
from services.provider_factory import ProviderFactory

# Create service using configured provider
service = ProviderFactory.create_query_service()

# Or specify provider explicitly
gemini_service = ProviderFactory.create_query_service("gemini")
openai_service = ProviderFactory.create_query_service("openai")

# Generate queries
queries = service.generate_search_queries("machine learning", max_queries=10)
```

## Cost Comparison

| Metric | OpenAI (GPT-4o) | Gemini (2.5 Flash) | Savings |
|--------|-----------------|---------------------|---------|
| Input (per 1K tokens) | $2.50 | $0.075 | **33x cheaper** |
| Output (per 1K tokens) | $10.00 | $0.30 | **33x cheaper** |
| Embeddings (per 1K) | $0.020 | $0.025 | Similar |

**Monthly savings example**: For 1M tokens of query generation, Gemini costs $375 vs OpenAI's $12,500 - saving $12,125/month!

## Features Comparison

### OpenAI Advantages
- ✅ Proven performance and reliability
- ✅ Excellent structured JSON outputs
- ✅ Strong ecosystem support

### Gemini Advantages
- ✅ **33x cheaper** for query generation
- ✅ **2M token context** (vs 128K for GPT-4o)
- ✅ **Native thinking mode** for complex reasoning
- ✅ **Better multilingual support**
- ✅ **Faster for most operations**
- ✅ **Free tier available**

## Architecture

### File Structure
```
backend/src/
├── services/
│   ├── query_service.py          # OpenAI implementation
│   ├── gemini_query_service.py   # Gemini implementation  
│   └── provider_factory.py       # Abstraction layer
├── embeddings.py                 # OpenAI embeddings
├── gemini_embeddings.py          # Gemini embeddings
└── config.py                     # Provider configuration
```

### Provider Factory

The `ProviderFactory` handles provider selection and service creation:

```python
# Get current provider
current = ProviderFactory.get_current_provider()

# Compare providers
comparison = ProviderFactory.compare_providers()

# Get recommendation
recommended = ProviderFactory.recommend_provider(
    cost_sensitive=True,
    multilingual=False, 
    long_context=False
)
```

## Testing

### Test Individual Providers

```bash
# Test Gemini
python scripts/test_gemini_integration.py --test-provider gemini

# Test OpenAI  
python scripts/test_gemini_integration.py --test-provider openai
```

### Compare Performance

```bash
# Full comparison
python scripts/test_gemini_integration.py --compare

# Custom topic
python scripts/test_gemini_integration.py --compare --topic "neural networks"
```

### Check Configuration

```bash
# Verify API keys
python scripts/test_gemini_integration.py --check-keys

# Cost analysis
python scripts/test_gemini_integration.py --cost-analysis
```

## Migration Guide

### From OpenAI to Gemini

1. **Set environment variable**:
   ```bash
   export LLM_PROVIDER=gemini
   export GEMINI_API_KEY=your_key_here
   ```

2. **Test the integration**:
   ```bash
   python scripts/test_gemini_integration.py --test-provider gemini
   ```

3. **Generate test queries**:
   ```bash
   python scripts/gpt_query_generator.py "test topic" --provider gemini --preview
   ```

4. **Update your scripts** to use the provider factory

### Backward Compatibility

- ✅ Existing OpenAI embeddings are preserved
- ✅ Old configurations continue to work
- ✅ Gradual migration is supported
- ✅ Provider can be switched at runtime

## Troubleshooting

### Common Issues

1. **"Gemini API key not found"**
   - Set `GEMINI_API_KEY` environment variable
   - Check `.env` file exists and is loaded

2. **"Invalid provider"**  
   - Ensure `LLM_PROVIDER` is set to "openai" or "gemini"
   - Check provider spelling

3. **"Module not found: google.genai"**
   - Install the dependency: `pip install google-genai`
   - Ensure you're using the NEW unified SDK, not the deprecated one

4. **API quota exceeded**
   - Check your API usage on Google AI Studio
   - Gemini has generous free tier limits

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed API calls and responses
```

## Best Practices

### Provider Selection

- **Use Gemini** for:
  - Cost-sensitive applications
  - High-volume query generation  
  - Multilingual content
  - Long context processing

- **Use OpenAI** for:
  - Mission-critical applications requiring proven reliability
  - Complex structured outputs
  - Specific GPT model requirements

### Configuration

1. **Set provider in environment**, not in code
2. **Use the provider factory** for service creation
3. **Test both providers** in your setup
4. **Monitor costs** with both providers

### Performance

1. **Cache embeddings** aggressively (both providers)
2. **Batch requests** when possible
3. **Use appropriate models** for each task
4. **Monitor rate limits** 

## Next Steps

1. **Install dependencies**: `pip install google-genai`
2. **Get API key**: Visit [Google AI Studio](https://ai.google.dev/)
3. **Configure environment**: Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`
4. **Test integration**: Run `python scripts/test_gemini_integration.py --compare`
5. **Start saving money**: Begin using Gemini for query generation

## Support

- **Gemini API Docs**: [ai.google.dev](https://ai.google.dev/gemini-api/docs)
- **OpenAI API Docs**: [platform.openai.com](https://platform.openai.com/docs)
- **Integration Issues**: Check the test script output for diagnostics