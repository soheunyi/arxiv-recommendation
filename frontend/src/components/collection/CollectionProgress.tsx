import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  CheckCircleIcon, 
  XCircleIcon, 
  ArrowPathIcon,
  DocumentTextIcon,
  ClockIcon 
} from '@heroicons/react/24/outline';

import { useAppDispatch, useAppSelector } from '@store';
import { fetchCollectionStatus, clearCollection } from '@store/slices/papersSlice';

interface CollectionProgressProps {
  className?: string;
}

export const CollectionProgress: React.FC<CollectionProgressProps> = ({ 
  className = '' 
}) => {
  const dispatch = useAppDispatch();
  const { collection } = useAppSelector((state) => state.papers);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Poll for status updates when collection is running
  useEffect(() => {
    if (collection.current && collection.current.status === 'running') {
      // Poll every 2 seconds
      intervalRef.current = setInterval(() => {
        dispatch(fetchCollectionStatus(collection.current!.id));
      }, 2000);
    } else {
      // Clear polling when not running
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [dispatch, collection.current?.status, collection.current?.id]);

  // Don't render if no collection
  if (!collection.current) {
    return null;
  }

  const { status, progress, papers_found, current_query, keyword, total_queries } = collection.current;

  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <ArrowPathIcon className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-secondary-400" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running':
        return 'border-blue-200 bg-blue-50';
      case 'completed':
        return 'border-green-200 bg-green-50';
      case 'failed':
        return 'border-red-200 bg-red-50';
      default:
        return 'border-secondary-200 bg-secondary-50';
    }
  };

  const getProgressBarColor = () => {
    switch (status) {
      case 'running':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-secondary-500';
    }
  };

  const handleDismiss = () => {
    dispatch(clearCollection());
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`rounded-lg border-2 p-4 ${getStatusColor()} ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3 flex-1">
          <div className="flex-shrink-0 mt-0.5">
            {getStatusIcon()}
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium text-secondary-900">
                Collecting papers: "{keyword}"
              </h3>
              
              {(status === 'completed' || status === 'failed') && (
                <button
                  onClick={handleDismiss}
                  className="text-secondary-400 hover:text-secondary-600 transition-colors"
                >
                  <XCircleIcon className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Progress Bar */}
            <div className="mb-3">
              <div className="flex items-center justify-between text-xs text-secondary-600 mb-1">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-secondary-200 rounded-full h-2">
                <motion.div
                  className={`h-2 rounded-full ${getProgressBarColor()}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>
            </div>

            {/* Status Details */}
            <div className="space-y-2">
              {current_query && (
                <div className="flex items-start space-x-2">
                  <DocumentTextIcon className="h-4 w-4 text-secondary-500 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-secondary-600 truncate">
                    {current_query}
                  </p>
                </div>
              )}

              <div className="flex items-center space-x-4 text-xs text-secondary-600">
                {papers_found > 0 && (
                  <span className="flex items-center space-x-1">
                    <DocumentTextIcon className="h-3 w-3" />
                    <span>{papers_found} papers found</span>
                  </span>
                )}
                
                {total_queries > 0 && (
                  <span className="flex items-center space-x-1">
                    <ArrowPathIcon className="h-3 w-3" />
                    <span>{total_queries} queries</span>
                  </span>
                )}
              </div>
            </div>

            {/* Status Message */}
            {status === 'completed' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 p-2 bg-green-100 border border-green-200 rounded text-xs text-green-700"
              >
                ✅ Collection completed successfully! Found {papers_found} papers.
              </motion.div>
            )}

            {status === 'failed' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 p-2 bg-red-100 border border-red-200 rounded text-xs text-red-700"
              >
                ❌ Collection failed. Please try again with a different keyword.
              </motion.div>
            )}

            {status === 'running' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-3 p-2 bg-blue-100 border border-blue-200 rounded text-xs text-blue-700"
              >
                🔄 Collection in progress... This may take 1-2 minutes.
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};