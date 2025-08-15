# Suggested Commands for Development

## Backend Development

### Running the Application
```bash
# Start both frontend and backend servers
uv run python start_servers.py

# Run backend API only
uv run uvicorn backend.api:app --reload --port 8000

# Run main CLI application
uv run python main.py run
```

### Database Operations
```bash
# Setup database (first time)
uv run python scripts/setup_database.py

# Run migrations
uv run python scripts/migrate_embeddings.py

# Check cache statistics
uv run python scripts/check_cache_stats.py
```

### Testing
```bash
# Run backend tests
uv run pytest tests/

# Run with coverage
uv run pytest --cov=backend --cov-report=term-missing

# Test core pipeline
uv run python scripts/test_pipeline.py
```

### Code Quality
```bash
# Run linter
uv run ruff check backend/

# Auto-fix linting issues
uv run ruff check --fix backend/

# Format code
uv run ruff format backend/

# Type checking
uv run mypy backend/
```

## Frontend Development

### Running the Application
```bash
# Start development server
cd frontend && npm run dev

# Build for production
cd frontend && npm run build

# Preview production build
cd frontend && npm run preview
```

### Code Quality
```bash
# Run linter
cd frontend && npm run lint

# Fix linting issues
cd frontend && npm run lint:fix

# Type checking
cd frontend && npm run type-check
```

### Testing
```bash
# Run tests
cd frontend && npm run test

# Run tests with UI
cd frontend && npm run test:ui
```

## Package Management

### Python (Backend)
```bash
# Install all dependencies
uv sync

# Install with dev dependencies
uv sync --group dev

# Install with web dependencies
uv sync --group web

# Add new dependency
uv add <package-name>

# Update dependencies
uv sync --upgrade
```

### Node.js (Frontend)
```bash
# Install dependencies
cd frontend && npm install

# Add new dependency
cd frontend && npm install <package-name>

# Add dev dependency
cd frontend && npm install -D <package-name>
```

## Git Commands
```bash
# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "feat: description"

# Push to remote
git push origin <branch-name>
```

## System Utilities (macOS/Darwin)
```bash
# List files
ls -la

# Find files
find . -name "*.py"

# Search in files
grep -r "pattern" .

# Process monitoring
ps aux | grep python

# Port usage
lsof -i :8000
```