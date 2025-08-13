import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FunnelIcon,
  MagnifyingGlassIcon,
  AdjustmentsHorizontalIcon,
  SparklesIcon,
  ChevronDownIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store';
import { fetchPapers } from '@store/slices/papersSlice';
import { updateRating, fetchUserRatings } from '@store/slices/ratingsSlice';
import { PaperCard } from '@components/papers/PaperCard';
import { LoadingSpinner } from '@components/common/LoadingSpinner';

type FilterCategory = 'all' | 'unrated' | 'rated' | 'favorites';
type SortOption = 'date' | 'title' | 'rating' | 'relevance';

export const RatingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  
  // Redux state
  const { 
    items: papers, 
    loading: papersLoading 
  } = useAppSelector((state) => state.papers);
  
  // Additional state for total paper count
  const [totalPapersCount, setTotalPapersCount] = useState<number>(0);
  
  const { 
    userRatings: ratings,
    loading: ratingsLoading 
  } = useAppSelector((state) => state.ratings);
  
  // Local state
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState<FilterCategory>('all');
  const [sortBy, setSortBy] = useState<SortOption>('date');
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(1);
  const [papersPerPage] = useState(10);
  
  // Load data on component mount
  useEffect(() => {
    const loadData = async () => {
      // First get the total count from stats
      try {
        const statsResponse = await fetch('/api/papers/stats');
        const statsData = await statsResponse.json();
        if (statsData.success && statsData.data) {
          setTotalPapersCount(statsData.data.total);
        }
      } catch (error) {
        console.error('Failed to fetch paper stats:', error);
      }
      
      // Then fetch papers for rating interface
      dispatch(fetchPapers({ limit: 50 })); // Still fetch 50 for display, but show correct total
      dispatch(fetchUserRatings());
    };
    
    loadData();
  }, [dispatch]);
  
  // Filter and sort papers
  const filteredPapers = React.useMemo(() => {
    if (!papers || !Array.isArray(papers)) return [];
    let filtered = [...papers];
    
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(paper => 
        paper.title.toLowerCase().includes(query) ||
        paper.abstract.toLowerCase().includes(query) ||
        paper.authors.some(author => author.toLowerCase().includes(query))
      );
    }
    
    // Category filter
    switch (filterCategory) {
      case 'unrated':
        filtered = filtered.filter(paper => !ratings[paper.id]);
        break;
      case 'rated':
        filtered = filtered.filter(paper => ratings[paper.id]);
        break;
      case 'favorites':
        filtered = filtered.filter(paper => ratings[paper.id]?.rating >= 4);
        break;
    }
    
    // Sort papers
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'title':
          return a.title.localeCompare(b.title);
        case 'rating':
          const ratingA = ratings[a.id]?.rating || 0;
          const ratingB = ratings[b.id]?.rating || 0;
          return ratingB - ratingA;
        case 'relevance':
          return (b.similarity_score || 0) - (a.similarity_score || 0);
        case 'date':
        default:
          return new Date(b.published_date).getTime() - new Date(a.published_date).getTime();
      }
    });
    
    return filtered;
  }, [papers, ratings, searchQuery, filterCategory, sortBy]);
  
  // Paginated papers
  const paginatedPapers = React.useMemo(() => {
    const startIndex = (page - 1) * papersPerPage;
    return filteredPapers.slice(startIndex, startIndex + papersPerPage);
  }, [filteredPapers, page, papersPerPage]);
  
  const totalPages = Math.ceil(filteredPapers.length / papersPerPage);
  
  // Event handlers
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
  
  const handleRefresh = async () => {
    // Refresh total count
    try {
      const statsResponse = await fetch('/api/papers/stats');
      const statsData = await statsResponse.json();
      if (statsData.success && statsData.data) {
        setTotalPapersCount(statsData.data.total);
      }
    } catch (error) {
      console.error('Failed to fetch paper stats:', error);
    }
    
    dispatch(fetchPapers({ limit: 50 }));
    dispatch(fetchUserRatings());
    toast.success('Papers refreshed!');
  };
  
  const clearFilters = () => {
    setSearchQuery('');
    setFilterCategory('all');
    setSortBy('date');
    setPage(1);
  };
  
  // Stats
  const ratedCount = ratings ? Object.keys(ratings).length : 0;
  const unratedCount = totalPapersCount - ratedCount;
  const favoritesCount = ratings ? Object.values(ratings).filter(r => r.rating >= 4).length : 0;
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">⭐ Rate Papers</h1>
          <p className="mt-2 text-secondary-600">
            Rate papers to improve your recommendations! {ratedCount} rated of {totalPapersCount} total papers
          </p>
        </div>
        
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleRefresh}
          disabled={papersLoading.papers}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
        >
          {papersLoading.papers ? (
            <LoadingSpinner size="sm" />
          ) : (
            <SparklesIcon className="h-4 w-4" />
          )}
          Refresh Papers
        </motion.button>
      </div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border border-secondary-200">
          <div className="text-sm font-medium text-secondary-600">Total Papers</div>
          <div className="text-2xl font-bold text-secondary-900">{totalPapersCount}</div>
        </div>
        <div className="bg-white p-4 rounded-lg border border-secondary-200">
          <div className="text-sm font-medium text-secondary-600">Rated</div>
          <div className="text-2xl font-bold text-green-600">{ratedCount}</div>
        </div>
        <div className="bg-white p-4 rounded-lg border border-secondary-200">
          <div className="text-sm font-medium text-secondary-600">Unrated</div>
          <div className="text-2xl font-bold text-orange-600">{unratedCount}</div>
        </div>
        <div className="bg-white p-4 rounded-lg border border-secondary-200">
          <div className="text-sm font-medium text-secondary-600">Favorites (4+ stars)</div>
          <div className="text-2xl font-bold text-yellow-600">{favoritesCount}</div>
        </div>
      </div>
      
      {/* Search and Filters */}
      <div className="bg-white p-6 rounded-lg border border-secondary-200">
        <div className="flex flex-col lg:flex-row gap-4 items-stretch lg:items-center">
          {/* Search */}
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
            <input
              type="text"
              placeholder="Search papers by title, abstract, or authors..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          
          {/* Category Filter */}
          <div className="relative">
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value as FilterCategory)}
              className="appearance-none bg-white border border-secondary-300 rounded-lg px-4 py-2.5 pr-8 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="all">All Papers</option>
              <option value="unrated">Unrated</option>
              <option value="rated">Rated</option>
              <option value="favorites">Favorites (4+ stars)</option>
            </select>
            <ChevronDownIcon className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400 pointer-events-none" />
          </div>
          
          {/* Sort */}
          <div className="relative">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as SortOption)}
              className="appearance-none bg-white border border-secondary-300 rounded-lg px-4 py-2.5 pr-8 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="date">Sort by Date</option>
              <option value="title">Sort by Title</option>
              <option value="rating">Sort by Rating</option>
              <option value="relevance">Sort by Relevance</option>
            </select>
            <ChevronDownIcon className="absolute right-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400 pointer-events-none" />
          </div>
          
          {/* Clear Filters */}
          {(searchQuery || filterCategory !== 'all' || sortBy !== 'date') && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-2 px-3 py-2 text-secondary-600 hover:text-secondary-900 transition-colors"
            >
              <XMarkIcon className="h-4 w-4" />
              Clear
            </button>
          )}
        </div>
        
        {/* Results count */}
        <div className="mt-4 text-sm text-secondary-600">
          Showing {paginatedPapers.length} of {filteredPapers.length} papers
        </div>
      </div>
      
      {/* Papers List */}
      {papersLoading.papers ? (
        <div className="text-center py-12">
          <LoadingSpinner />
          <p className="mt-4 text-secondary-600">Loading papers...</p>
        </div>
      ) : paginatedPapers.length > 0 ? (
        <div className="space-y-4">
          {paginatedPapers.map((paper, index) => (
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
        <div className="text-center py-12">
          <FunnelIcon className="mx-auto h-12 w-12 text-secondary-400" />
          <h3 className="mt-2 text-sm font-medium text-secondary-900">
            No papers found
          </h3>
          <p className="mt-1 text-sm text-secondary-500">
            Try adjusting your search or filter criteria.
          </p>
        </div>
      )}
      
      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-secondary-600">
            Page {page} of {totalPages}
          </div>
          
          <div className="flex gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="px-3 py-2 border border-secondary-300 rounded-lg disabled:opacity-50 hover:bg-secondary-50 transition-colors"
            >
              Previous
            </button>
            
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="px-3 py-2 border border-secondary-300 rounded-lg disabled:opacity-50 hover:bg-secondary-50 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};