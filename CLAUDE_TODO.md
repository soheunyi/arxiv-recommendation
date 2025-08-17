 - [x] Debug (500 error)
 - [x] Automatic paper collection every morning
 - [x] Automatic update of scores every day (requires recomputing user preference embedding and scoring all papers -- use cached embeddings)
     - [x] Should implement a collection logic to only look for recent papers.
 - [x] One problem is that the user preference embedding does not reflect the user's "recent" preferences. There should be a logic to keep track of the user's recent preferences and update the embedding accordingly --- maybe "all time" embedding and "recent" embedding?
     - [x] Maybe we can think of weighting the "recent" embedding by the user's recent preferences? EMA style?
 - [x] Implement a HARD logic for collecting papers to avoid sending multiple requests to openai API.
 - [x] Use cached embeddings for scoring papers.


- Reference tracking algorithm: (1) parse html to get references (2) google each paper and cross-validate arxiv id (3) fetch with arxiv api

## Development Roadmap

### **Priority 1: Quality & Reliability Foundation** 🔧 *(1-2 weeks)*
- [ ] **Fix Reference System Test Failures**
  - [ ] Resolve 13 failing tests in reference extraction and OpenAlex integration
  - [ ] Fix missing `integration_database` fixture in performance tests
  - [ ] Improve OpenAlex API error handling and rate limiting
- [ ] **Improve Test Coverage & CI/CD**
  - [ ] Increase coverage from 19% to 60%+ for production readiness
  - [ ] Add comprehensive integration tests for two-stage reference workflow
  - [ ] Implement performance benchmarks and monitoring
- [ ] **API Reliability Improvements**
  - [ ] Add proper error handling for all 40+ API endpoints
  - [ ] Implement rate limiting and request validation
  - [ ] Add API response caching for better performance

### **Priority 2: Enhanced User Experience** 🎨 *(2-3 weeks)*
- [ ] **Smart Query Enhancement**
  - [ ] GPT-based query refinement with upload date filtering
  - [ ] Intelligent query expansion using user's rating history
  - [ ] Query suggestion system based on user preferences
- [ ] **Reference System UX Optimization**
  - [ ] Batch reference fetching for multiple papers
  - [ ] Citation network visualization
  - [ ] Similar paper discovery based on reference patterns
  - [ ] "Paper trail" navigation for following citation chains
- [ ] **Progressive Web App Features**
  - [ ] Offline reading capability for downloaded papers
  - [ ] Push notifications for new recommendations
  - [ ] Mobile-optimized layouts and gestures

### **Priority 3: Advanced Intelligence Features** 🧠 *(3-4 weeks)*
- [ ] **Web Search Integration**
  - [ ] LLM-powered pre-arXiv web search for query context
  - [ ] Trending topic detection from academic news
  - [ ] Research area discovery through web analysis
  - [ ] Context-aware paper recommendations
- [ ] **Enhanced Recommendation Algorithm**
  - [ ] Topic modeling with web-informed categories
  - [ ] Temporal trend analysis for emerging fields
  - [ ] Collaborative filtering with similar researchers
  - [ ] Multi-modal recommendations (papers + code + datasets)
- [ ] **Research Assistant Features**
  - [ ] Paper summarization with key insights extraction
  - [ ] Research gap identification across paper collections
  - [ ] Automated literature review generation
  - [ ] Research timeline visualization

### **Priority 4: Production Readiness** 🚀 *(2-3 weeks)*
- [ ] **Infrastructure & Deployment**
  - [ ] Docker containers for backend and frontend
  - [ ] Docker-compose for development environment
  - [ ] Production deployment with environment management
- [ ] **Monitoring & Analytics**
  - [ ] Comprehensive application monitoring
  - [ ] User behavior analytics
  - [ ] System health dashboards
  - [ ] Cost tracking and alerting
- [ ] **Security & Compliance**
  - [ ] Authentication and authorization
  - [ ] API key management and rotation
  - [ ] Backup and disaster recovery procedures
  - [ ] Privacy controls and GDPR compliance

## Completed Features ✅
- [x] **Two-stage reference tracking system** (ArXiv HTML parsing + OpenAlex enrichment)
  - [x] Stage 1: Immediate ArXiv HTML parsing for instant citation discovery
  - [x] Stage 2: OpenAlex API enrichment for enhanced metadata (automated)
  - [x] Hybrid service orchestrating both stages with fallback support
  - [x] Database schema with OpenAlex fields (citation counts, topics, etc.)
  - [x] Batch processing for scheduler integration at 3:00 AM UTC
  - [x] Comprehensive test suite (Unit: 18/18 passed, Integration: 4/6 passed)

## Success Metrics 📊
- **Quality**: Test coverage 19% → 75%+, API error rate <1%
- **Performance**: Page load <2s, Reference fetching <5s
- **Cost**: Maintain 33x savings with Gemini optimization
- **User Experience**: >4.5/5 satisfaction, +50% session duration