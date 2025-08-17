import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon,
  InformationCircleIcon,
  FunnelIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { useAppDispatch, useAppSelector } from '../../store';
import toast from 'react-hot-toast';

import { Paper } from '../../types';
import { papersService } from '../../services/papersService';
import { PaperCard } from './PaperCard';
import { PaperScoreDisplay, ScoreExplanation } from './PaperScoreDisplay';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { updateRating } from '../../store/slices/ratingsSlice';

interface ScoredPapersViewProps {
  limit?: number;
  showExplanation?: boolean;
  allowFiltering?: boolean;
}

type ScoreFilter = 'all' | 'excellent' | 'good' | 'fair' | 'poor';

export const ScoredPapersView: React.FC<ScoredPapersViewProps> = ({
  limit = 20,
  showExplanation = true,
  allowFiltering = true
}) => {
  const dispatch = useAppDispatch();
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scoreFilter, setScoreFilter] = useState<ScoreFilter>('all');
  const [showExplanationPanel, setShowExplanationPanel] = useState(showExplanation);

  const loadScoredPapers = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await papersService.getPapersByScore(limit);
      
      // Filter papers that actually have scores
      const scoredPapers = data.filter(paper => 
        paper.current_score !== null && 
        paper.current_score !== undefined && 
        !isNaN(paper.current_score)
      );
      
      setPapers(scoredPapers);
      
      if (scoredPapers.length === 0) {
        setError('No scored papers available. Please collect some papers first using the "Collect Papers" page, then generate recommendations to see relevance scores.');
      }
    } catch (err) {
      console.error('Error loading scored papers:', err);
      setError('Failed to load scored papers. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadScoredPapers();
  }, [limit]);

  const handleRatingChange = async (paperId: string, rating: number) => {
    try {
      await dispatch(updateRating({ paper_id: paperId, rating })).unwrap();
      toast.success(`Rated ${rating}/5 stars!`);
    } catch (error: any) {
      toast.error(error.message || 'Failed to update rating');
      throw error;
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

  // Filter papers based on score range
  const filteredPapers = papers.filter(paper => {
    if (scoreFilter === 'all') return true;
    
    const score = paper.current_score || 0;
    switch (scoreFilter) {
      case 'excellent': return score >= 0.8;
      case 'good': return score >= 0.65 && score < 0.8;
      case 'fair': return score >= 0.5 && score < 0.65;
      case 'poor': return score < 0.5;
      default: return true;
    }
  });

  // Calculate score statistics
  const scoreStats = papers.reduce((stats, paper) => {
    const score = paper.current_score || 0;
    if (score >= 0.8) stats.excellent++;
    else if (score >= 0.65) stats.good++;
    else if (score >= 0.5) stats.fair++;
    else stats.poor++;
    stats.total++;
    return stats;
  }, { excellent: 0, good: 0, fair: 0, poor: 0, total: 0 });

  const averageScore = papers.length > 0 
    ? papers.reduce((sum, paper) => sum + (paper.current_score || 0), 0) / papers.length 
    : 0;

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-center py-8">
          <LoadingSpinner size="md" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center py-8">
          <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={loadScoredPapers}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            <ArrowPathIcon className="h-4 w-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with stats */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-secondary-900 flex items-center gap-2">
            <ChartBarIcon className="h-6 w-6 text-primary-600" />
            Scored Papers ({papers.length})
          </h2>
          
          <div className="flex items-center gap-2">
            {allowFiltering && (
              <select
                value={scoreFilter}
                onChange={(e) => setScoreFilter(e.target.value as ScoreFilter)}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
              >
                <option value="all">All Scores</option>
                <option value="excellent">Excellent (≥0.8)</option>
                <option value="good">Good (0.65-0.79)</option>
                <option value="fair">Fair (0.5-0.64)</option>
                <option value="poor">Poor (&lt;0.5)</option>
              </select>
            )}
            
            <button
              onClick={() => setShowExplanationPanel(!showExplanationPanel)}
              className="p-2 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-md transition-colors"
              title="Toggle explanation"
            >
              <InformationCircleIcon className="h-5 w-5" />
            </button>
            
            <button
              onClick={loadScoredPapers}
              className="p-2 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-md transition-colors"
              title="Refresh"
            >
              <ArrowPathIcon className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Score statistics */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{scoreStats.total}</div>
            <div className="text-sm text-gray-600">Total Papers</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{scoreStats.excellent}</div>
            <div className="text-sm text-gray-600">Excellent</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{scoreStats.good}</div>
            <div className="text-sm text-gray-600">Good</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">{scoreStats.fair}</div>
            <div className="text-sm text-gray-600">Fair</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{scoreStats.poor}</div>
            <div className="text-sm text-gray-600">Poor</div>
          </div>
        </div>

        {/* Average score display */}
        <div className="border-t pt-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700">Average Relevance Score:</span>
            <PaperScoreDisplay score={averageScore} compact={true} showTooltip={false} />
          </div>
        </div>
      </div>

      {/* Score explanation panel */}
      {showExplanationPanel && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-white rounded-lg shadow p-6"
        >
          <ScoreExplanation />
        </motion.div>
      )}

      {/* Papers list */}
      <div className="space-y-4">
        {filteredPapers.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-center">
            <FunnelIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No papers match the selected score filter.</p>
            <button
              onClick={() => setScoreFilter('all')}
              className="mt-2 text-primary-600 hover:text-primary-700"
            >
              Show all papers
            </button>
          </div>
        ) : (
          filteredPapers.map((paper, index) => (
            <motion.div
              key={paper.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <PaperCard
                paper={paper}
                onRatingChange={handleRatingChange}
                onNotesChange={handleNotesChange}
                showNotes={true}
                className="shadow-sm hover:shadow-md transition-shadow"
              />
            </motion.div>
          ))
        )}
      </div>

      {/* Load more section */}
      {papers.length >= limit && (
        <div className="text-center">
          <p className="text-sm text-gray-500 mb-4">
            Showing {papers.length} papers. Want to see more?
          </p>
          <button
            onClick={() => loadScoredPapers()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Generate More Recommendations
          </button>
        </div>
      )}
    </div>
  );
};