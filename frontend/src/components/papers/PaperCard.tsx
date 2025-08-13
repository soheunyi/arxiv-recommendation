import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  DocumentTextIcon, 
  LinkIcon, 
  CalendarIcon,
  TagIcon,
  UserGroupIcon 
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

import { Paper } from '@types';
import { StarRating } from '@components/rating/StarRating';
import { AbstractRenderer } from './AbstractRenderer';

interface PaperCardProps {
  paper: Paper;
  onRatingChange?: (paperId: string, rating: number) => void;
  onNotesChange?: (paperId: string, notes: string) => void;
  compact?: boolean;
  showNotes?: boolean;
  className?: string;
}

export const PaperCard: React.FC<PaperCardProps> = ({
  paper,
  onRatingChange,
  onNotesChange,
  compact = false,
  showNotes = false,
  className,
}) => {
  const [showFullAbstract, setShowFullAbstract] = useState(false);
  const [notes, setNotes] = useState(paper.notes || '');
  const [isNotesExpanded, setIsNotesExpanded] = useState(false);
  
  const handleRatingChange = (rating: number) => {
    onRatingChange?.(paper.id, rating);
  };
  
  const handleNotesSubmit = () => {
    if (notes !== paper.notes) {
      onNotesChange?.(paper.id, notes);
    }
    setIsNotesExpanded(false);
  };
  
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateString;
    }
  };
  
  const truncateAuthors = (authors: string[], maxLength: number = 3) => {
    if (authors.length <= maxLength) {
      return authors.join(', ');
    }
    return `${authors.slice(0, maxLength).join(', ')} +${authors.length - maxLength} more`;
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={clsx(
        'bg-white rounded-lg border border-secondary-200 hover:border-secondary-300 transition-colors duration-200',
        'hover:shadow-lg',
        {
          'p-4': compact,
          'p-6': !compact,
        },
        className
      )}
    >
      {/* Header */}
      <div className="flex justify-between items-start gap-4 mb-4">
        <div className="flex-1 min-w-0">
          <h3 className={clsx(
            'font-semibold text-secondary-900 leading-tight',
            {
              'text-lg': !compact,
              'text-base': compact,
            }
          )}>
            {paper.title}
          </h3>
          
          {/* Score badge */}
          {paper.current_score !== undefined && (
            <div className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800 mt-2">
              Score: {paper.current_score.toFixed(3)}
            </div>
          )}
        </div>
        
        {/* Actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {paper.pdf_url && (
            <a
              href={paper.pdf_url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-secondary-500 hover:text-primary-600 hover:bg-primary-50 rounded-md transition-colors"
              title="View PDF"
            >
              <DocumentTextIcon className="w-5 h-5" />
            </a>
          )}
          
          {paper.arxiv_url && (
            <a
              href={paper.arxiv_url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-secondary-500 hover:text-primary-600 hover:bg-primary-50 rounded-md transition-colors"
              title="View on ArXiv"
            >
              <LinkIcon className="w-5 h-5" />
            </a>
          )}
        </div>
      </div>
      
      {/* Metadata */}
      <div className={clsx(
        'flex flex-wrap items-center gap-4 text-sm text-secondary-600 mb-4',
        { 'gap-2 text-xs': compact }
      )}>
        {/* Authors */}
        <div className="flex items-center gap-1">
          <UserGroupIcon className="w-4 h-4" />
          <span title={paper.authors.join(', ')}>
            {truncateAuthors(paper.authors, compact ? 2 : 3)}
          </span>
        </div>
        
        {/* Category */}
        <div className="flex items-center gap-1">
          <TagIcon className="w-4 h-4" />
          <span className="px-2 py-1 bg-secondary-100 rounded-full text-xs">
            {paper.category}
          </span>
        </div>
        
        {/* Date */}
        <div className="flex items-center gap-1">
          <CalendarIcon className="w-4 h-4" />
          <span>{formatDate(paper.published_date)}</span>
        </div>
      </div>
      
      {/* Abstract */}
      {!compact && (
        <div className="mb-4">
          <AbstractRenderer
            abstract={paper.abstract}
            expanded={showFullAbstract}
            onToggle={() => setShowFullAbstract(!showFullAbstract)}
            maxLength={300}
          />
        </div>
      )}
      
      {/* Rating */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-secondary-700">Rate:</span>
          <StarRating
            rating={paper.rating || 0}
            interactive
            clearable
            size={compact ? 'sm' : 'md'}
            onChange={handleRatingChange}
          />
        </div>
        
        {/* Notes toggle */}
        {showNotes && (
          <button
            onClick={() => setIsNotesExpanded(!isNotesExpanded)}
            className="text-sm text-primary-600 hover:text-primary-700 font-medium"
          >
            {isNotesExpanded ? 'Hide Notes' : 'Add Notes'}
          </button>
        )}
      </div>
      
      {/* Notes section */}
      {showNotes && isNotesExpanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-4 pt-4 border-t border-secondary-200"
        >
          <label className="block text-sm font-medium text-secondary-700 mb-2">
            Notes
          </label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            className="w-full px-3 py-2 border border-secondary-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
            rows={3}
            placeholder="Add your notes about this paper..."
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={() => {
                setNotes(paper.notes || '');
                setIsNotesExpanded(false);
              }}
              className="px-3 py-1 text-sm text-secondary-600 hover:text-secondary-700"
            >
              Cancel
            </button>
            <button
              onClick={handleNotesSubmit}
              className="px-3 py-1 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
            >
              Save
            </button>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
};