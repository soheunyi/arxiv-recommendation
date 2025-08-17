import React from 'react';
import { motion } from 'framer-motion';
import { ChartBarIcon } from '@heroicons/react/24/outline';

import { ScoredPapersView } from '@components/papers/ScoredPapersView';

export const ScoredPapersPage: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900 flex items-center gap-3">
          <ChartBarIcon className="h-8 w-8 text-primary-600" />
          Relevance Scores
        </h1>
        <p className="mt-2 text-secondary-600">
          View papers ranked by AI-calculated relevance scores based on your preferences and rating history.
        </p>
      </div>

      {/* Scored papers view */}
      <ScoredPapersView 
        limit={50}
        showExplanation={true}
        allowFiltering={true}
      />
    </motion.div>
  );
};