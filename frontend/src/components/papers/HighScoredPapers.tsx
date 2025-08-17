import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { TrophyIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { PaperCard } from './PaperCard';
import { LoadingSpinner } from '@components/common/LoadingSpinner';
import { papersService } from '@services/papersService';
import { type Paper } from '../../types';

interface HighScoredPapersProps {
  limit?: number;
  showTitle?: boolean;
  compact?: boolean;
}

export const HighScoredPapers: React.FC<HighScoredPapersProps> = ({
  limit = 10,
  showTitle = true,
  compact = false
}) => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHighScoredPapers = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await papersService.getPapersByScore(limit);
      // Filter papers that actually have scores
      const scoredPapers = data.filter(paper => 
        paper.current_score !== null && paper.current_score !== undefined
      );
      setPapers(scoredPapers);
    } catch (err) {
      console.error('Error loading high-scored papers:', err);
      setError('Failed to load high-scored papers');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHighScoredPapers();
  }, [limit]);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        {showTitle && (
          <h2 className="text-xl font-bold text-secondary-900 mb-4 flex items-center gap-2">
            <TrophyIcon className="h-6 w-6 text-yellow-500" />
            Top Scored Papers
          </h2>
        )}
        <div className="flex justify-center py-8">
          <LoadingSpinner size="md" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        {showTitle && (
          <h2 className="text-xl font-bold text-secondary-900 mb-4 flex items-center gap-2">
            <TrophyIcon className="h-6 w-6 text-yellow-500" />
            Top Scored Papers
          </h2>
        )}
        <div className="text-center py-8">
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={loadHighScoredPapers}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            <ArrowPathIcon className="h-4 w-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (papers.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        {showTitle && (
          <h2 className="text-xl font-bold text-secondary-900 mb-4 flex items-center gap-2">
            <TrophyIcon className="h-6 w-6 text-yellow-500" />
            Top Scored Papers
          </h2>
        )}
        <div className="text-center py-8">
          <TrophyIcon className="h-12 w-12 text-secondary-300 mx-auto mb-4" />
          <p className="text-secondary-600">No scored papers available yet.</p>
          <p className="text-sm text-secondary-500 mt-2">
            Recommendation scores will appear here once generated.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {showTitle && (
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-secondary-900 flex items-center gap-2">
            <TrophyIcon className="h-6 w-6 text-yellow-500" />
            Top Scored Papers
          </h2>
          <button
            onClick={loadHighScoredPapers}
            className="p-2 text-secondary-500 hover:text-secondary-700 hover:bg-secondary-100 rounded-md"
            title="Refresh"
          >
            <ArrowPathIcon className="h-5 w-5" />
          </button>
        </div>
      )}
      
      <div className="space-y-4">
        {papers.map((paper, index) => (
          <motion.div
            key={paper.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="relative"
          >
            {/* Ranking badge */}
            <div className="absolute -left-2 -top-2 z-10 w-8 h-8 bg-yellow-500 text-white rounded-full flex items-center justify-center text-sm font-bold shadow-lg">
              {index + 1}
            </div>
            
            <PaperCard
              paper={paper}
              compact={compact}
              className="border-l-4 border-l-yellow-400"
            />
          </motion.div>
        ))}
      </div>
      
      {papers.length >= limit && (
        <div className="text-center mt-6">
          <p className="text-sm text-secondary-500">
            Showing top {limit} scored papers
          </p>
        </div>
      )}
    </div>
  );
};