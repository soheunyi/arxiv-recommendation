import React from 'react';
import { motion } from 'framer-motion';
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface PaperScoreDisplayProps {
  score: number;
  compact?: boolean;
  showTooltip?: boolean;
  className?: string;
}

export const PaperScoreDisplay: React.FC<PaperScoreDisplayProps> = ({
  score,
  compact = false,
  showTooltip = true,
  className
}) => {
  // Convert score to percentage (0-1 range to 0-100%)
  const percentage = Math.max(0, Math.min(100, score * 100));
  
  // Determine score level and styling
  const getScoreLevel = (score: number) => {
    if (score >= 0.8) return { level: 'excellent', color: 'green', label: 'Excellent Match' };
    if (score >= 0.65) return { level: 'good', color: 'blue', label: 'Good Match' };
    if (score >= 0.5) return { level: 'fair', color: 'yellow', label: 'Fair Match' };
    if (score >= 0.3) return { level: 'poor', color: 'orange', label: 'Poor Match' };
    return { level: 'very-poor', color: 'red', label: 'Very Poor Match' };
  };

  const scoreInfo = getScoreLevel(score);

  const getColorClasses = (color: string) => {
    const colorMap = {
      green: {
        bg: 'bg-green-100',
        text: 'text-green-800',
        border: 'border-green-200',
        progress: 'bg-green-500',
        progressBg: 'bg-green-100'
      },
      blue: {
        bg: 'bg-blue-100',
        text: 'text-blue-800',
        border: 'border-blue-200',
        progress: 'bg-blue-500',
        progressBg: 'bg-blue-100'
      },
      yellow: {
        bg: 'bg-yellow-100',
        text: 'text-yellow-800',
        border: 'border-yellow-200',
        progress: 'bg-yellow-500',
        progressBg: 'bg-yellow-100'
      },
      orange: {
        bg: 'bg-orange-100',
        text: 'text-orange-800',
        border: 'border-orange-200',
        progress: 'bg-orange-500',
        progressBg: 'bg-orange-100'
      },
      red: {
        bg: 'bg-red-100',
        text: 'text-red-800',
        border: 'border-red-200',
        progress: 'bg-red-500',
        progressBg: 'bg-red-100'
      }
    };
    return colorMap[color as keyof typeof colorMap];
  };

  const colors = getColorClasses(scoreInfo.color);

  if (compact) {
    return (
      <div 
        className={clsx(
          'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border',
          colors.bg,
          colors.text,
          colors.border,
          className
        )}
        title={showTooltip ? `${scoreInfo.label}: ${score.toFixed(3)} (${percentage.toFixed(1)}%)` : undefined}
      >
        <div className="w-2 h-2 rounded-full bg-current opacity-70" />
        {score.toFixed(3)}
      </div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={clsx(
        'inline-flex items-center gap-2 px-3 py-2 rounded-lg border',
        colors.bg,
        colors.border,
        className
      )}
    >
      <div className="flex items-center gap-2">
        <div className="text-sm font-medium">
          <span className={clsx('text-xs font-normal', colors.text)}>Relevance Score</span>
          <div className={clsx('font-bold', colors.text)}>
            {score.toFixed(3)}
          </div>
        </div>
        
        {/* Progress bar */}
        <div className="flex flex-col items-end">
          <div className={clsx('w-16 h-2 rounded-full', colors.progressBg)}>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${percentage}%` }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className={clsx('h-full rounded-full', colors.progress)}
            />
          </div>
          <span className={clsx('text-xs mt-1', colors.text)}>
            {percentage.toFixed(0)}%
          </span>
        </div>
      </div>

      {showTooltip && (
        <div className="group relative">
          <InformationCircleIcon className={clsx('w-4 h-4 cursor-help', colors.text)} />
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap z-10 pointer-events-none">
            <div className="font-medium">{scoreInfo.label}</div>
            <div className="text-gray-300">
              Based on content similarity to your preferences
            </div>
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
          </div>
        </div>
      )}
    </motion.div>
  );
};

// Component for displaying score explanation
export const ScoreExplanation: React.FC = () => {
  const scoreRanges = [
    { range: '0.80 - 1.00', level: 'Excellent Match', color: 'green', description: 'Highly relevant to your interests' },
    { range: '0.65 - 0.79', level: 'Good Match', color: 'blue', description: 'Well-aligned with your preferences' },
    { range: '0.50 - 0.64', level: 'Fair Match', color: 'yellow', description: 'Moderately relevant' },
    { range: '0.30 - 0.49', level: 'Poor Match', color: 'orange', description: 'Limited relevance' },
    { range: '0.00 - 0.29', level: 'Very Poor Match', color: 'red', description: 'Not well-matched to your interests' }
  ];

  return (
    <div className="bg-gray-50 rounded-lg p-4 space-y-3">
      <h4 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
        <InformationCircleIcon className="w-4 h-4" />
        Relevance Score Guide
      </h4>
      <div className="space-y-2">
        {scoreRanges.map((range, index) => (
          <div key={index} className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <div className={clsx(
                'w-3 h-3 rounded-full',
                range.color === 'green' && 'bg-green-500',
                range.color === 'blue' && 'bg-blue-500',
                range.color === 'yellow' && 'bg-yellow-500',
                range.color === 'orange' && 'bg-orange-500',
                range.color === 'red' && 'bg-red-500'
              )} />
              <span className="font-medium">{range.level}</span>
              <span className="text-gray-500">({range.range})</span>
            </div>
            <span className="text-gray-600">{range.description}</span>
          </div>
        ))}
      </div>
      <div className="text-xs text-gray-500 pt-2 border-t border-gray-200">
        Scores are calculated using AI-powered content similarity analysis based on your rating history and preferences.
      </div>
    </div>
  );
};