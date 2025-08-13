1. Debug (500 error)
2. Automatic paper collection every morning
3. Automatic update of scores every day (requires recomputing user preference embedding and scoring all papers -- use cached embeddings)
   1. Should implement a collection logic to only look for recent papers.
4. One problem is that the user preference embedding does not reflect the user's "recent" preferences. There should be a logic to keep track of the user's recent preferences and update the embedding accordingly --- maybe "all time" embedding and "recent" embedding?
   1. Maybe we can think of weighting the "recent" embedding by the user's recent preferences? EMA style?
5. Implement a HARD logic for collecting papers to avoid sending multiple requests to openai API.
6. Use cached embeddings for scoring papers.