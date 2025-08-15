# Code Style and Conventions

## Python Backend Conventions

### General Style
- **Python Version**: 3.11+ with modern type hints
- **Line Length**: 88 characters (Black/Ruff standard)
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Double quotes preferred
- **Imports**: Sorted with isort rules (stdlib, third-party, local)

### Naming Conventions
- **Classes**: PascalCase (e.g., `DatabaseManager`, `ArXivClient`)
- **Functions/Methods**: snake_case (e.g., `fetch_papers`, `generate_embeddings`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `BASE_URL`, `RATE_LIMIT_DELAY`)
- **Private Methods**: Leading underscore (e.g., `_create_tables`)
- **Async Functions**: Prefix with async, use await consistently

### Type Hints
- Always use type hints for function signatures
- Use `Optional[]` for nullable types
- Use `List[]`, `Dict[]`, `Tuple[]` from typing
- Use dataclasses for structured data

### Documentation
- Triple-quoted docstrings for all public classes and methods
- Include Args, Returns, and Raises sections
- Example:
```python
"""
Brief description of function.

Args:
    param1: Description of param1
    param2: Description of param2
    
Returns:
    Description of return value
    
Raises:
    ExceptionType: When this happens
"""
```

### Error Handling
- Use specific exception types
- Always log errors with context
- Provide meaningful error messages
- Use try/except/finally blocks appropriately

## TypeScript/React Frontend Conventions

### General Style
- **TypeScript**: Strict mode enabled
- **React**: Functional components with hooks
- **Formatting**: Prettier with 2-space indentation
- **Semicolons**: Not required (omitted)

### Naming Conventions
- **Components**: PascalCase (e.g., `PaperCard`, `SearchPage`)
- **Files**: PascalCase for components, camelCase for utilities
- **Interfaces/Types**: PascalCase with 'I' or 'T' prefix optional
- **Props Interfaces**: ComponentNameProps (e.g., `PaperCardProps`)
- **Hooks**: camelCase with 'use' prefix (e.g., `useAppSelector`)

### File Organization
```
src/
  components/     # Reusable UI components
  pages/         # Page-level components
  services/      # API services
  store/         # Redux store and slices
  types/         # TypeScript type definitions
  utils/         # Utility functions
  hooks/         # Custom React hooks
```

### Component Structure
```tsx
import React from 'react'
// imports...

interface ComponentNameProps {
  // props definition
}

export const ComponentName: React.FC<ComponentNameProps> = ({ prop1, prop2 }) => {
  // hooks
  // state
  // effects
  // handlers
  
  return (
    // JSX
  )
}
```

### State Management
- Use Redux Toolkit for global state
- Use React Query for server state
- Local state with useState/useReducer
- Custom hooks for reusable logic

## Database Conventions

### Table Naming
- Plural snake_case (e.g., `papers`, `user_ratings`)
- Junction tables: table1_table2 (e.g., `paper_references`)

### Column Naming
- snake_case for all columns
- Foreign keys: table_id (e.g., `paper_id`)
- Timestamps: created_at, updated_at
- Booleans: is_active, has_been_processed

## API Conventions

### Endpoints
- RESTful naming: /api/resource
- Use HTTP verbs appropriately (GET, POST, PUT, DELETE)
- Plural resource names (e.g., /api/papers)
- Nested resources when logical (e.g., /api/papers/{id}/references)

### Response Format
```json
{
  "data": {},
  "success": true,
  "message": "Optional message",
  "timestamp": "ISO 8601 timestamp"
}
```

### Error Responses
```json
{
  "data": null,
  "success": false,
  "message": "Error description",
  "error_id": "Unique error identifier",
  "timestamp": "ISO 8601 timestamp"
}
```