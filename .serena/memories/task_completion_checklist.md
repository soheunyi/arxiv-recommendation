# Task Completion Checklist

When completing any development task, follow these steps:

## 1. Code Quality Checks

### Python Backend
```bash
# Run linter
uv run ruff check backend/

# Fix any linting issues
uv run ruff check --fix backend/

# Format code
uv run ruff format backend/

# Type checking
uv run mypy backend/
```

### TypeScript Frontend
```bash
# Type checking
cd frontend && npm run type-check

# Lint check
cd frontend && npm run lint

# Fix linting issues
cd frontend && npm run lint:fix
```

## 2. Testing

### Backend Tests
```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest --cov=backend --cov-report=term-missing

# Test specific functionality
uv run python scripts/test_pipeline.py
```

### Frontend Tests
```bash
cd frontend && npm run test
```

## 3. Documentation Updates

- Update relevant docstrings
- Update README.md if adding new features
- Update API_REFERENCE.md for new endpoints
- Add inline comments for complex logic

## 4. Database Migrations

If database schema changed:
```bash
# Check current schema
uv run python scripts/setup_database.py

# Run any migration scripts
uv run python scripts/migrate_embeddings.py
```

## 5. Configuration Updates

- Update .env.example if new environment variables added
- Update config.py for new configuration options
- Document configuration changes

## 6. API Testing

For new/modified endpoints:
```bash
# Start the API server
uv run uvicorn backend.api:app --reload

# Test endpoints manually or with curl
curl http://localhost:8000/api/endpoint
```

## 7. Integration Testing

```bash
# Test full pipeline
uv run python scripts/test_pipeline.py

# Test with both frontend and backend
uv run python start_servers.py
```

## 8. Git Commit

```bash
# Stage changes
git add .

# Check what will be committed
git status

# Commit with descriptive message
git commit -m "feat: add reference tracking system"
```

## 9. Final Verification

- [ ] Code passes all linting checks
- [ ] All tests pass
- [ ] Type checking passes
- [ ] Documentation is updated
- [ ] No console.log or print statements left
- [ ] Error handling is comprehensive
- [ ] API responses follow standard format
- [ ] Database migrations work correctly
- [ ] Frontend and backend integrate properly

## 10. Performance Considerations

- Check for N+1 query problems
- Verify rate limiting is respected
- Ensure async operations are properly handled
- Monitor memory usage for large datasets
- Verify caching is working correctly