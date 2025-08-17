# Features

1. Make a feature to manually enter arXiv id to one that is not found when searching arXiv.
2. Refine the arxiv query process
   - If not found, hand arxiv query to gpt-4o-mini to refine the query
   - re-search with the refined query
   - use structured output -- json
3. Use DuckDuckGo to search for the paper if not found in arXiv