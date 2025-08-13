import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ArrowPathIcon, 
  ChartBarIcon, 
  SparklesIcon 
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store';
import { fetchRecommendations, generateRecommendations } from '@store/slices/papersSlice';
import { updateRating } from '@store/slices/ratingsSlice';
import { fetchCacheStats } from '@store/slices/systemSlice';
import { PaperCard } from '@components/papers/PaperCard';
import { LoadingSpinner } from '@components/common/LoadingSpinner';

export const DashboardPage: React.FC = () => {
  const dispatch = useAppDispatch();
  
  const { 
    recommendations, 
    loading: papersLoading 
  } = useAppSelector((state) => state.papers);
  
  const { 
    cache, 
    loading: systemLoading 
  } = useAppSelector((state) => state.system);
  
  const { 
    loading: ratingsLoading 
  } = useAppSelector((state) => state.ratings);
  
  useEffect(() => {
    // Load initial data
    dispatch(fetchRecommendations());
    dispatch(fetchCacheStats());
  }, [dispatch]);
  
  const handleGenerateRecommendations = async () => {
    try {
      await dispatch(generateRecommendations()).unwrap();
      toast.success('New recommendations generated successfully!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to generate recommendations');
    }
  };
  
  const handleViewCacheStats = () => {
    if (cache) {
      toast.success(
        `Cache Hit Rate: ${(cache.hit_rate * 100).toFixed(1)}%\n` +
        `Total Embeddings: ${cache.total_embeddings}\n` +
        `Cache Size: ${cache.cache_size_mb.toFixed(2)} MB\n` +
        `Daily Cost: $${cache.daily_cost.toFixed(6)}`,
        { duration: 6000 }
      );
    }
  };
  
  const handleRatingChange = async (paperId: string, rating: number) => {
    try {
      await dispatch(updateRating({ paper_id: paperId, rating })).unwrap();
      toast.success(`Rated ${rating}/5 stars!`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to update rating');
    }
  };
  
  const handleNotesChange = async (paperId: string, notes: string) => {
    try {
      await dispatch(updateRating({ paper_id: paperId, rating: 0, notes })).unwrap();
      toast.success('Notes saved!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to save notes');
    }
  };
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">
          📚 ArXiv Recommendation Dashboard
        </h1>
        <p className="mt-2 text-secondary-600">
          Welcome to your personalized research paper recommendation system!
        </p>
      </div>
      
      {/* Action buttons */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleGenerateRecommendations}
          disabled={papersLoading.recommendations}
          className="flex items-center justify-center gap-2 rounded-lg bg-primary-600 px-4 py-3 text-white font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {papersLoading.recommendations ? (
            <LoadingSpinner size="sm" />
          ) : (
            <ArrowPathIcon className="h-5 w-5" />
          )}
          Generate New Recommendations
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleViewCacheStats}
          disabled={systemLoading.cache}
          className="flex items-center justify-center gap-2 rounded-lg bg-secondary-600 px-4 py-3 text-white font-medium hover:bg-secondary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {systemLoading.cache ? (
            <LoadingSpinner size="sm" />
          ) : (
            <ChartBarIcon className="h-5 w-5" />
          )}
          View Cache Stats
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => toast('🎉 Analytics coming soon!')}
          className="flex items-center justify-center gap-2 rounded-lg bg-accent-600 px-4 py-3 text-white font-medium hover:bg-accent-700 transition-colors"
        >
          <SparklesIcon className="h-5 w-5" />
          Show Analytics
        </motion.button>
      </div>
      
      {/* Cache stats display */}
      {cache && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div className="rounded-lg bg-white p-4 border border-secondary-200">
            <div className="text-sm font-medium text-secondary-600">Cache Hit Rate</div>
            <div className="text-2xl font-bold text-secondary-900">
              {(cache.hit_rate * 100).toFixed(1)}%
            </div>
          </div>
          
          <div className="rounded-lg bg-white p-4 border border-secondary-200">
            <div className="text-sm font-medium text-secondary-600">Total Embeddings</div>
            <div className="text-2xl font-bold text-secondary-900">
              {cache.total_embeddings.toLocaleString()}
            </div>
          </div>
          
          <div className="rounded-lg bg-white p-4 border border-secondary-200">
            <div className="text-sm font-medium text-secondary-600">Cache Size</div>
            <div className="text-2xl font-bold text-secondary-900">
              {cache.cache_size_mb.toFixed(2)} MB
            </div>
          </div>
          
          <div className="rounded-lg bg-white p-4 border border-secondary-200">
            <div className="text-sm font-medium text-secondary-600">Daily Cost</div>
            <div className="text-2xl font-bold text-secondary-900">
              ${cache.daily_cost.toFixed(6)}
            </div>
          </div>
        </div>
      )}
      
      {/* Recommendations section */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-secondary-900">
            🎯 Today's Recommendations
          </h2>
          
          {papersLoading.recommendations && (
            <div className="flex items-center gap-2 text-sm text-secondary-600">
              <LoadingSpinner size="sm" />
              Loading recommendations...
            </div>
          )}
        </div>
        
        {recommendations.length > 0 ? (
          <div className="space-y-4">
            {recommendations.slice(0, 10).map((paper, index) => (
              <motion.div
                key={paper.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-center gap-4 mb-2">
                  <div className="flex-shrink-0 w-8 h-8 bg-primary-100 text-primary-800 rounded-full flex items-center justify-center font-semibold text-sm">
                    #{index + 1}
                  </div>
                </div>
                
                <PaperCard
                  paper={paper}
                  onRatingChange={handleRatingChange}
                  onNotesChange={handleNotesChange}
                  showNotes
                />
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <SparklesIcon className="mx-auto h-12 w-12 text-secondary-400" />
            <h3 className="mt-2 text-sm font-medium text-secondary-900">
              No recommendations available
            </h3>
            <p className="mt-1 text-sm text-secondary-500">
              Click 'Generate New Recommendations' to get started!
            </p>
          </div>
        )}
      </div>
    </div>
  );
};