import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  BookOpenIcon,
  FunnelIcon,
  ChevronDownIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store';
import { fetchPapers } from '@store/slices/papersSlice';
import { updateRating } from '@store/slices/ratingsSlice';
import { PaperCard } from '@components/papers/PaperCard';
import { LoadingSpinner } from '@components/common/LoadingSpinner';

export const PapersPage: React.FC = () => {
  const dispatch = useAppDispatch();
  
  const { 
    items: papers,
    loading: papersLoading 
  } = useAppSelector((state) => state.papers);
  
  const { 
    userRatings: ratings,
    loading: ratingsLoading 
  } = useAppSelector((state) => state.ratings);
  
  // Local state for filters
  const [categoryFilter, setCategoryFilter] = useState<string>('All');
  const [ratingFilter, setRatingFilter] = useState<string>('All');
  const [sortBy, setSortBy] = useState<'date' | 'title' | 'rating'>('date');
  const [page, setPage] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  
  // Available categories (this could come from the API)
  const categories = [
    'All',
    'cs.AI',
    'cs.LG', 
    'cs.CV',
    'cs.CL',
    'cs.CR',
    'cs.DC',
    'cs.DS',
    'cs.HC',
    'cs.IR',
    'cs.SE',
    'stat.ML',
    'math.ST',
    'physics.data-an'
  ];
  
  const ratingOptions = ['All', 'Unrated', '1', '2', '3', '4', '5'];
  
  useEffect(() => {
    loadPapers();
  }, [dispatch, page, categoryFilter, ratingFilter]);
  
  const loadPapers = async () => {
    try {
      await dispatch(fetchPapers({
        page,
        limit: 20,
        filters: {
          category: categoryFilter === 'All' ? undefined : categoryFilter,
          rating: ratingFilter === 'All' ? undefined : (ratingFilter === 'Unrated' ? 'unrated' : parseInt(ratingFilter))
        }
      })).unwrap();
    } catch (error: any) {
      toast.error(error.message || 'Failed to load papers');
    }
  };
  
  const handleRefresh = () => {
    setPage(1);
    loadPapers();
    toast.success('Papers refreshed!');
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
  
  // Sort papers locally
  const sortedPapers = [...papers].sort((a, b) => {
    switch (sortBy) {
      case 'title':
        return a.title.localeCompare(b.title);
      case 'rating':
        const ratingA = ratings[a.id]?.rating || 0;
        const ratingB = ratings[b.id]?.rating || 0;
        return ratingB - ratingA;
      case 'date':
      default:
        return new Date(b.published_date).getTime() - new Date(a.published_date).getTime();
    }
  });
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">
            📄 All Papers
          </h1>
          <p className="mt-2 text-secondary-600">
            Browse and manage all papers in your database
          </p>
        </div>
        
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleRefresh}
          disabled={papersLoading.fetch}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {papersLoading.fetch ? (
            <LoadingSpinner size="sm" />
          ) : (
            <ArrowPathIcon className="h-4 w-4" />
          )}
          Refresh
        </motion.button>
      </div>
      
      {/* Filters */}
      <div className="bg-white p-4 rounded-lg border border-secondary-200 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center gap-2 text-secondary-600 hover:text-secondary-900 transition-colors"
          >
            <FunnelIcon className="h-4 w-4" />
            Filters & Sorting
            <ChevronDownIcon className={`h-4 w-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
          </button>
          
          <div className="text-sm text-secondary-600">
            {papers.length} papers
          </div>
        </div>
        
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-secondary-200"
          >
            {/* Category Filter */}
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">Category</label>
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="w-full p-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                {categories.map(category => (
                  <option key={category} value={category}>{category}</option>
                ))}
              </select>
            </div>
            
            {/* Rating Filter */}
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">Rating</label>
              <select
                value={ratingFilter}
                onChange={(e) => setRatingFilter(e.target.value)}
                className="w-full p-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                {ratingOptions.map(rating => (
                  <option key={rating} value={rating}>{rating}</option>
                ))}
              </select>
            </div>
            
            {/* Sort By */}
            <div>
              <label className="block text-sm font-medium text-secondary-700 mb-2">Sort By</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'date' | 'title' | 'rating')}
                className="w-full p-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500"
              >
                <option value="date">Date (Newest First)</option>
                <option value="title">Title (A-Z)</option>
                <option value="rating">Rating (Highest First)</option>
              </select>
            </div>
          </motion.div>
        )}
      </div>
      
      {/* Papers List */}
      {papersLoading.fetch ? (
        <div className="text-center py-12">
          <LoadingSpinner />
          <p className="mt-4 text-secondary-600">Loading papers...</p>
        </div>
      ) : sortedPapers.length > 0 ? (
        <div className="space-y-4">
          {sortedPapers.map((paper, index) => (
            <motion.div
              key={paper.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
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
          <BookOpenIcon className="mx-auto h-12 w-12 text-secondary-400" />
          <h3 className="mt-2 text-sm font-medium text-secondary-900">
            No papers found
          </h3>
          <p className="mt-1 text-sm text-secondary-500">
            {categoryFilter !== 'All' || ratingFilter !== 'All' 
              ? 'Try adjusting your filters.'
              : 'No papers available in the database.'
            }
          </p>
        </div>
      )}
    </div>
  );
};