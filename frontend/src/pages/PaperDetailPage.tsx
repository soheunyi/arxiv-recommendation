import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  DocumentTextIcon, 
  LinkIcon, 
  CalendarIcon,
  TagIcon,
  UserGroupIcon,
  ArrowLeftIcon,
  BookOpenIcon
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

import { Paper } from '../types/paper';
import { StarRating } from '@components/rating/StarRating';
import { AbstractRenderer } from '@components/papers/AbstractRenderer';
import { PaperReferences } from '@components/papers/PaperReferences';
import { PaperScoreDisplay } from '@components/papers/PaperScoreDisplay';
import { LoadingSpinner } from '@components/common/LoadingSpinner';
import { papersService } from '../services';
import { ratingsService } from '../services';

export const PaperDetailPage: React.FC = () => {
  const { paperId } = useParams<{ paperId: string }>();
  const navigate = useNavigate();
  
  const [paper, setPaper] = useState<Paper | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'references' | 'citations'>('overview');

  useEffect(() => {
    const fetchPaper = async () => {
      if (!paperId) return;
      
      try {
        setLoading(true);
        setError(null);
        
        const paper = await papersService.getPaper(paperId);
        setPaper(paper);
      } catch (err) {
        console.error('Failed to fetch paper:', err);
        setError('Failed to load paper details. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchPaper();
  }, [paperId]);

  const handleRatingChange = async (paperId: string, rating: number) => {
    try {
      await ratingsService.updateRating({ paper_id: paperId, rating });
      // Update local state
      if (paper) {
        setPaper({ ...paper, rating });
      }
    } catch (error) {
      console.error('Failed to update rating:', error);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error || !paper) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            {error || 'Paper not found'}
          </h1>
          <button
            onClick={() => navigate('/papers')}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
          >
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Back to Papers
          </button>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-6xl mx-auto px-4 py-8"
    >
      {/* Header with navigation */}
      <div className="mb-8">
        <nav className="flex items-center space-x-2 text-sm text-gray-500 mb-4">
          <Link to="/papers" className="hover:text-primary-600">
            Papers
          </Link>
          <span>/</span>
          <span className="text-gray-900 font-medium">Paper Details</span>
        </nav>
        
        <button
          onClick={() => navigate(-1)}
          className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
        >
          <ArrowLeftIcon className="w-4 h-4 mr-2" />
          Back
        </button>
      </div>

      {/* Paper content */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
        {/* Header section */}
        <div className="p-8 border-b border-gray-200">
          <div className="flex justify-between items-start gap-6">
            <div className="flex-1 min-w-0">
              <h1 className="text-3xl font-bold text-gray-900 leading-tight mb-4">
                {paper.title}
              </h1>
              
              {/* Enhanced score display */}
              {paper.current_score !== undefined && paper.current_score !== null && (
                <div className="mb-4">
                  <PaperScoreDisplay 
                    score={paper.current_score} 
                    compact={false}
                    showTooltip={true}
                  />
                </div>
              )}
              
              {/* Metadata */}
              <div className="flex flex-wrap items-center gap-6 text-gray-600 mb-6">
                {/* Authors */}
                <div className="flex items-center gap-2">
                  <UserGroupIcon className="w-5 h-5" />
                  <span className="font-medium">
                    {paper.authors.join(', ')}
                  </span>
                </div>
                
                {/* Category */}
                <div className="flex items-center gap-2">
                  <TagIcon className="w-5 h-5" />
                  <span className="px-3 py-1 bg-gray-100 rounded-full text-sm font-medium">
                    {paper.category}
                  </span>
                </div>
                
                {/* Date */}
                <div className="flex items-center gap-2">
                  <CalendarIcon className="w-5 h-5" />
                  <span>{formatDate(paper.published_date)}</span>
                </div>
              </div>

              {/* Rating */}
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium text-gray-700">Your Rating:</span>
                <StarRating
                  rating={paper.rating ?? 0}
                  interactive
                  clearable
                  size="md"
                  onChange={(rating) => handleRatingChange(paper.id, rating)}
                />
              </div>
            </div>
            
            {/* Actions */}
            <div className="flex flex-col gap-3 flex-shrink-0">
              {paper.pdf_url && (
                <a
                  href={paper.pdf_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
                >
                  <DocumentTextIcon className="w-4 h-4 mr-2" />
                  View PDF
                </a>
              )}
              
              {paper.arxiv_url && (
                <a
                  href={paper.arxiv_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  <LinkIcon className="w-4 h-4 mr-2" />
                  View on ArXiv
                </a>
              )}
            </div>
          </div>
        </div>

        {/* Tab navigation */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-8">
            {[
              { key: 'overview', label: 'Overview', icon: BookOpenIcon },
              { key: 'references', label: 'References', icon: LinkIcon },
              { key: 'citations', label: 'Citations', icon: DocumentTextIcon }
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as any)}
                className={clsx(
                  'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm',
                  activeTab === key
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                )}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab content */}
        <div className="p-8">
          {activeTab === 'overview' && (
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Abstract</h2>
              <AbstractRenderer
                abstract={paper.abstract}
                expanded={true}
                onToggle={() => {}}
                maxLength={Number.MAX_SAFE_INTEGER}
              />
              
              {paper.notes && (
                <div className="mt-8">
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Your Notes</h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-700 whitespace-pre-wrap">{paper.notes}</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'references' && (
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-6">References</h2>
              <PaperReferences
                paperId={paper.rowid ? paper.rowid.toString() : paper.id}
                paperTitle={paper.title}
              />
            </div>
          )}

          {activeTab === 'citations' && (
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Citations</h2>
              <p className="text-gray-500">
                Papers that cite this work will be displayed here. This feature requires citation network data.
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};