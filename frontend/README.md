# ArXiv Recommendation System - Frontend

A modern React frontend for the ArXiv Recommendation System, built with TypeScript, Redux Toolkit, and Tailwind CSS.

## Features

- 📚 **Dashboard**: View personalized paper recommendations
- ⭐ **Rating System**: Interactive star ratings with auto-save
- 🔍 **Search**: Advanced paper search and filtering
- 📊 **Analytics**: Usage statistics and insights
- ⚙️ **Settings**: System configuration and preferences
- 🎨 **Modern UI**: Responsive design with Tailwind CSS
- 🔄 **Real-time Updates**: Live data synchronization
- 📱 **Mobile Friendly**: Responsive across all devices
- ♿ **Accessible**: WCAG 2.1 AA compliant

## Tech Stack

### Core
- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and dev server

### State Management
- **Redux Toolkit** - Modern Redux with RTK Query
- **React Query** - Server state management and caching

### UI & Styling
- **Tailwind CSS** - Utility-first CSS framework
- **Headless UI** - Unstyled, accessible UI components
- **Heroicons** - Beautiful hand-crafted SVG icons
- **Framer Motion** - Smooth animations and transitions

### Math & Rendering
- **KaTeX** - Fast math typesetting for LaTeX
- **React KaTeX** - React wrapper for KaTeX

### Development
- **ESLint** - Code linting and formatting
- **Prettier** - Code formatting
- **Vitest** - Unit testing framework

## Project Structure

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/        # Reusable React components
│   │   ├── common/       # Layout, navigation, shared components
│   │   ├── papers/       # Paper-related components
│   │   ├── rating/       # Rating system components
│   │   ├── dashboard/    # Dashboard-specific components
│   │   ├── analytics/    # Analytics and charts
│   │   └── settings/     # Settings page components
│   ├── pages/            # Page-level components
│   ├── hooks/            # Custom React hooks
│   ├── services/         # API service layer
│   ├── store/            # Redux store and slices
│   ├── types/            # TypeScript type definitions
│   ├── utils/            # Utility functions
│   └── styles/           # Global styles and themes
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── vite.config.ts
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on port specified by `BACKEND_PORT` (default: 8000)

### Installation

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser**:
   Navigate to `http://localhost:${FRONTEND_PORT}` (default: 3000)

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint errors
- `npm run type-check` - Run TypeScript compiler check
- `npm test` - Run tests
- `npm run test:ui` - Run tests with UI

## Component Architecture

### Core Components

#### StarRating
Interactive star rating component with hover effects, keyboard navigation, and accessibility support.

```tsx
<StarRating
  rating={4}
  interactive
  clearable
  onChange={(rating) => console.log(rating)}
/>
```

#### PaperCard
Comprehensive paper display with abstract rendering, rating interface, and action buttons.

```tsx
<PaperCard
  paper={paper}
  onRatingChange={handleRating}
  showNotes
  compact={false}
/>
```

#### AbstractRenderer
LaTeX-aware abstract renderer with math equation support using KaTeX.

```tsx
<AbstractRenderer
  abstract={paper.abstract}
  expanded={false}
  onToggle={() => setExpanded(!expanded)}
/>
```

### State Management

The application uses Redux Toolkit with the following store slices:

- **Papers**: Paper data, search results, recommendations
- **Ratings**: User ratings, optimistic updates, analytics
- **User**: Preferences, settings, session data
- **System**: Cache stats, notifications, online status

### API Integration

Service layer provides type-safe API communication:

```typescript
// Example API usage
const papers = await papersService.getPapers({
  page: 1,
  limit: 20,
  filters: { category: 'cs.AI' }
});

const rating = await ratingsService.updateRating({
  paper_id: 'paper123',
  rating: 5,
  notes: 'Excellent paper!'
});
```

## Styling & Theming

### Tailwind Configuration

Custom color palette optimized for scientific content:

- **Primary**: Blue tones for main actions
- **Secondary**: Gray tones for text and backgrounds  
- **Accent**: Purple tones for highlights
- **Rating**: Gold tones for star ratings

### Custom Components

Pre-built component classes for consistency:

```css
.btn-primary     /* Primary button styling */
.card           /* Paper card container */
.form-input     /* Form input styling */
.nav-link       /* Navigation link styling */
```

### Responsive Design

Mobile-first approach with breakpoints:

- `sm`: 640px and up
- `md`: 768px and up  
- `lg`: 1024px and up
- `xl`: 1280px and up

## Performance Optimizations

### Code Splitting
- Route-based splitting with React.lazy()
- Manual chunks for vendor libraries
- Dynamic imports for large components

### Caching Strategy
- React Query for API response caching
- Service worker for offline support
- Local storage for user preferences

### Bundle Optimization
- Tree shaking for unused code
- Image optimization and lazy loading
- Font preloading for critical text

## Accessibility

### WCAG 2.1 AA Compliance
- Keyboard navigation support
- Screen reader compatibility  
- High contrast mode support
- Focus management and indicators

### Inclusive Design
- Color blind friendly palette
- Reduced motion preferences
- Clear typography hierarchy
- Descriptive alt text and labels

## Development Guidelines

### Code Style
- Use TypeScript for type safety
- Follow React best practices and hooks patterns
- Implement proper error boundaries
- Write descriptive component and prop types

### Testing Strategy
- Unit tests for utilities and hooks
- Integration tests for API services
- Component tests for user interactions
- E2E tests for critical user flows

### Performance
- Implement React.memo for expensive components
- Use useCallback and useMemo appropriately
- Optimize re-renders with proper dependency arrays
- Monitor bundle size and loading times

## Backend Integration

### API Endpoints

The frontend expects the following REST API endpoints:

```
GET  /api/papers              # Get papers with pagination
GET  /api/papers/recent       # Get recent papers
POST /api/papers/search       # Search papers
GET  /api/recommendations     # Get recommendations
POST /api/recommendations/generate  # Generate new recommendations
POST /api/ratings/update      # Update paper rating
GET  /api/ratings/stats       # Get rating statistics
GET  /api/system/cache/stats  # Get cache statistics
POST /api/system/cache/clear  # Clear cache
```

### Data Format

Example API response format:

```json
{
  "data": {
    "papers": [...],
    "total": 100,
    "page": 1,
    "hasMore": true
  },
  "success": true,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Deployment

### Build for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

### Environment Variables

Create `.env.local` for local development:

```env
FRONTEND_PORT=3000
BACKEND_PORT=8000
```

### Hosting Options

The built application can be deployed to:

- **Vercel** - Recommended for React apps
- **Netlify** - Good for static hosting
- **AWS S3 + CloudFront** - Scalable option
- **Docker** - Containerized deployment

## Contributing

1. Follow the established code style and patterns
2. Write TypeScript types for all new interfaces
3. Add unit tests for new functionality
4. Update documentation for significant changes
5. Ensure accessibility compliance for UI changes

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

This project is part of the ArXiv Recommendation System and follows the same license as the main project.