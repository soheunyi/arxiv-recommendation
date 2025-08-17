import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MagnifyingGlassIcon,
  FunnelIcon,
  AdjustmentsHorizontalIcon,
  CalendarIcon,
  TagIcon,
  UserIcon,
  BookOpenIcon,
  XMarkIcon,
  ChevronDownIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store/index';
import { searchPapers, clearSearchResults } from '@store/slices/papersSlice';
import { updateRating, fetchUserRatings } from '@store/slices/ratingsSlice';
import { PaperCard } from '@components/papers/PaperCard';
import { LoadingSpinner } from '@components/common/LoadingSpinner';

type SearchScope = 'all' | 'title' | 'abstract' | 'authors';
type DateRange = 'all' | 'week' | 'month' | 'year' | 'custom';
type SortOption = 'relevance' | 'date' | 'title' | 'citations';

interface SearchFilters {
  scope: SearchScope;
  dateRange: DateRange;
  customDateFrom?: string;
  customDateTo?: string;
  categories: string[];
  minCitations?: number;
  sortBy: SortOption;
}

const ARXIV_CATEGORIES = [
  { id: 'cs.AI', name: 'Artificial Intelligence' },
  { id: 'cs.LG', name: 'Machine Learning' },
  { id: 'cs.CV', name: 'Computer Vision' },
  { id: 'cs.CL', name: 'Natural Language Processing' },
  { id: 'cs.CR', name: 'Cryptography' },
  { id: 'cs.DC', name: 'Distributed Computing' },
  { id: 'cs.DS', name: 'Data Structures and Algorithms' },
  { id: 'cs.HC', name: 'Human-Computer Interaction' },
  { id: 'cs.IR', name: 'Information Retrieval' },
  { id: 'cs.SE', name: 'Software Engineering' },
  { id: 'stat.ML', name: 'Statistics - Machine Learning' },
  { id: 'math.ST', name: 'Statistics Theory' },
  { id: 'physics.data-an', name: 'Data Analysis' },
];

export const SearchPage: React.FC = () => {
  const dispatch = useAppDispatch();
  
  // Redux state
  const { 
    searchResults, 
    loading: papersLoading 
  } = useAppSelector((state) => state.papers);
  
  const { 
    userRatings: ratings,
    loading: ratingsLoading 
  } = useAppSelector((state) => state.ratings);
  
  // Local state
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({
    scope: 'all',
    dateRange: 'all',
    categories: [],
    sortBy: 'relevance'
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [page, setPage] = useState(1);
  const [hasSearched, setHasSearched] = useState(false);
  
  // Load user ratings on mount
  useEffect(() => {
    dispatch(fetchUserRatings());
  }, [dispatch]);
  
  // Search handler
  const handleSearch = async (newQuery?: string, newFilters?: SearchFilters, newPage?: number) => {
    const searchQuery = newQuery ?? query;
    const searchFilters = newFilters ?? filters;
    const searchPage = newPage ?? page;
    
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }
    
    try {
      await dispatch(searchPapers({
        query: searchQuery,
        scope: searchFilters.scope,
        dateRange: searchFilters.dateRange,
        categories: searchFilters.categories,
        sortBy: searchFilters.sortBy,
        page: searchPage,
        limit: 20
      })).unwrap();
      
      setHasSearched(true);
      toast.success(`Found results for "${searchQuery}"`);
    } catch (error: any) {
      toast.error(error.message || 'Search failed');
    }
  };
  
  // Quick search suggestions
  const quickSearches = [
    'machine learning',
    'neural networks',
    'computer vision',
    'natural language processing',
    'deep learning',
    'reinforcement learning'
  ];
  
  // Event handlers
  const handleQuickSearch = (searchTerm: string) => {
    setQuery(searchTerm);
    handleSearch(searchTerm);
  };
  
  const handleFilterChange = (key: keyof SearchFilters, value: any) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    
    // Auto-search if we already have a query
    if (query.trim() && hasSearched) {
      handleSearch(query, newFilters, 1);
      setPage(1);
    }
  };
  
  const handleCategoryToggle = (categoryId: string) => {
    const newCategories = filters.categories.includes(categoryId)
      ? filters.categories.filter(id => id !== categoryId)
      : [...filters.categories, categoryId];
    
    handleFilterChange('categories', newCategories);
  };
  
  const clearSearch = () => {
    setQuery('');
    setFilters({
      scope: 'all',
      dateRange: 'all',
      categories: [],
      sortBy: 'relevance'
    });
    setPage(1);
    setHasSearched(false);
    dispatch(clearSearchResults());
  };
  
  const handleRatingChange = async (paperId: string, rating: number) => {
    try {
      await dispatch(updateRating({ paper_id: paperId, rating })).unwrap();
      toast.success(`Rated ${rating}/5 stars!`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to update rating');
      throw error; // Re-throw for component rollback
    }
  };
  
  const handleNotesChange = async (paperId: string, notes: string) => {
    try {
      await dispatch(updateRating({ 
        paper_id: paperId, 
        rating: ratings[paperId]?.rating || 0, 
        notes 
      })).unwrap();
      toast.success('Notes saved!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to save notes');
    }
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">🔍 Search Papers</h1>
        <p className="mt-2 text-secondary-600">
          Search and discover research papers from arXiv and other sources.
        </p>
      </div>
      
      {/* Search Interface */}
      <div className="bg-white p-6 rounded-lg border border-secondary-200 shadow-sm">
        {/* Main Search */}
        <div className="space-y-4">
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-secondary-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search for papers... (e.g., 'machine learning attention mechanisms')"
                className="w-full pl-10 pr-4 py-3 text-lg border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => handleSearch()}
              disabled={papersLoading.search || !query.trim()}
              className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {papersLoading.search ? (
                <LoadingSpinner size="sm" />
              ) : (
                'Search'
              )}
            </motion.button>
          </div>
          
          {/* Quick Search Suggestions */}
          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-secondary-600 py-2">Quick searches:</span>
            {quickSearches.map((term) => (
              <button
                key={term}
                onClick={() => handleQuickSearch(term)}
                className="px-3 py-1 text-sm bg-secondary-100 hover:bg-secondary-200 text-secondary-700 rounded-full transition-colors"
              >
                {term}
              </button>
            ))}
          </div>
        </div>
        
        {/* Advanced Filters Toggle */}
        <div className="mt-4 pt-4 border-t border-secondary-200">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-secondary-600 hover:text-secondary-900 transition-colors"
          >
            <AdjustmentsHorizontalIcon className="h-4 w-4" />
            Advanced Filters
            <ChevronDownIcon className={`h-4 w-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
          </button>
        </div>
        
        {/* Advanced Filters */}
        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 space-y-4 overflow-hidden"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Search Scope */}
                <div>
                  <label className="block text-sm font-medium text-secondary-700 mb-2">Search In</label>
                  <select
                    value={filters.scope}
                    onChange={(e) => handleFilterChange('scope', e.target.value as SearchScope)}
                    className="w-full p-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="all">All Fields</option>
                    <option value="title">Title Only</option>
                    <option value="abstract">Abstract Only</option>
                    <option value="authors">Authors Only</option>
                  </select>
                </div>
                
                {/* Date Range */}
                <div>
                  <label className="block text-sm font-medium text-secondary-700 mb-2">Date Range</label>
                  <select
                    value={filters.dateRange}
                    onChange={(e) => handleFilterChange('dateRange', e.target.value as DateRange)}
                    className="w-full p-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="all">All Time</option>
                    <option value="week">Past Week</option>
                    <option value="month">Past Month</option>
                    <option value="year">Past Year</option>
                  </select>
                </div>
                
                {/* Sort By */}
                <div>
                  <label className="block text-sm font-medium text-secondary-700 mb-2">Sort By</label>
                  <select
                    value={filters.sortBy}
                    onChange={(e) => handleFilterChange('sortBy', e.target.value as SortOption)}
                    className="w-full p-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="relevance">Relevance</option>
                    <option value="date">Newest First</option>
                    <option value="title">Title A-Z</option>
                    <option value="citations">Most Cited</option>
                  </select>
                </div>
              </div>
              
              {/* Categories */}
              <div>
                <label className="block text-sm font-medium text-secondary-700 mb-2">Categories</label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                  {ARXIV_CATEGORIES.map((category) => (
                    <label key={category.id} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={filters.categories.includes(category.id)}
                        onChange={() => handleCategoryToggle(category.id)}
                        className="mr-2 rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                      />
                      <span className="text-sm text-secondary-700">{category.name}</span>
                    </label>
                  ))}
                </div>
              </div>
              
              {/* Clear Filters */}
              {(filters.scope !== 'all' || filters.dateRange !== 'all' || filters.categories.length > 0 || filters.sortBy !== 'relevance') && (
                <button
                  onClick={() => setFilters({
                    scope: 'all',
                    dateRange: 'all',
                    categories: [],
                    sortBy: 'relevance'
                  })}
                  className="flex items-center gap-2 text-secondary-600 hover:text-secondary-900 transition-colors"
                >
                  <XMarkIcon className="h-4 w-4" />
                  Clear Filters
                </button>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      {/* Search Results */}
      {hasSearched && (
        <div className="space-y-4">
          {/* Results Header */}
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-secondary-900">
                Search Results {searchResults.length > 0 && `(${searchResults.length} found)`}
              </h2>
              {query && (
                <p className="text-secondary-600">
                  Results for: <span className="font-medium">"{query}"</span>
                </p>
              )}
            </div>
            
            {searchResults.length > 0 && (
              <button
                onClick={clearSearch}
                className="flex items-center gap-2 px-3 py-2 text-secondary-600 hover:text-secondary-900 transition-colors"
              >
                <XMarkIcon className="h-4 w-4" />
                Clear Search
              </button>
            )}
          </div>
          
          {/* Loading State */}
          {papersLoading.search ? (
            <div className="text-center py-12">
              <LoadingSpinner />
              <p className="mt-4 text-secondary-600">Searching papers...</p>
            </div>
          ) : searchResults.length > 0 ? (
            /* Results */
            <div className="space-y-4">
              {searchResults.map((paper, index) => (
                <motion.div
                  key={paper.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <PaperCard
                    paper={paper}
                    rating={ratings[paper.id]?.rating}
                    onRatingChange={handleRatingChange}
                    onNotesChange={handleNotesChange}
                    showNotes
                  />
                </motion.div>
              ))}
            </div>
          ) : (
            /* No Results */
            <div className="text-center py-12">
              <MagnifyingGlassIcon className="mx-auto h-12 w-12 text-secondary-400" />
              <h3 className="mt-2 text-sm font-medium text-secondary-900">
                No papers found
              </h3>
              <p className="mt-1 text-sm text-secondary-500">
                Try adjusting your search terms or filters.
              </p>
              <div className="mt-4">
                <button
                  onClick={() => setShowAdvanced(true)}
                  className="text-primary-600 hover:text-primary-700 font-medium"
                >
                  Try advanced filters
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Welcome State */}
      {!hasSearched && (
        <div className="text-center py-12">
          <div className="mx-auto w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-4">
            <SparklesIcon className="h-8 w-8 text-primary-600" />
          </div>
          <h3 className="text-lg font-medium text-secondary-900 mb-2">
            Ready to discover research papers?
          </h3>
          <p className="text-secondary-600 max-w-md mx-auto">
            Enter your search terms above to find relevant papers from arXiv and other academic sources. 
            Use advanced filters to refine your search.
          </p>
        </div>
      )}
    </div>
  );
};