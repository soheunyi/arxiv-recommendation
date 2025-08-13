import React, { useState, useEffect } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { 
  XMarkIcon, 
  ExclamationTriangleIcon,
  ShieldCheckIcon,
  ClockIcon,
  DocumentDuplicateIcon,
  TrashIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';

interface DatabaseCleanupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (options: CleanupOptions) => void;
  databaseStats?: {
    papers_count: number;
    user_ratings_count: number;
    paper_embeddings_count: number;
    search_history_count: number;
    recommendations_history_count: number;
  };
}

interface CleanupOptions {
  clean_papers: boolean;
  clean_ratings: boolean;
  clean_embeddings: boolean;
  clean_search_history: boolean;
  clean_recommendations: boolean;
  confirmation_text: string;
}

export const DatabaseCleanupModal: React.FC<DatabaseCleanupModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  databaseStats,
}) => {
  const [step, setStep] = useState<'warning' | 'options' | 'confirmation'>('warning');
  const [countdown, setCountdown] = useState(5);
  const [canProceed, setCanProceed] = useState(false);
  const [confirmationText, setConfirmationText] = useState('');
  const [cleanupOptions, setCleanupOptions] = useState<CleanupOptions>({
    clean_papers: true,
    clean_ratings: false, // Default to keeping ratings
    clean_embeddings: false, // Default to keeping expensive embeddings
    clean_search_history: true,
    clean_recommendations: true,
    confirmation_text: ''
  });

  const requiredText = 'DELETE ALL DATA';

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setStep('warning');
      setCountdown(5);
      setCanProceed(false);
      setConfirmationText('');
      setCleanupOptions({
        clean_papers: true,
        clean_ratings: false,
        clean_embeddings: false,
        clean_search_history: true,
        clean_recommendations: true,
        confirmation_text: ''
      });
    }
  }, [isOpen]);

  // Countdown timer for first step
  useEffect(() => {
    if (step === 'warning' && countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    } else if (step === 'warning' && countdown === 0) {
      setCanProceed(true);
    }
  }, [step, countdown]);

  const handleClose = () => {
    onClose();
  };

  const handleNextStep = () => {
    if (step === 'warning') {
      setStep('options');
    } else if (step === 'options') {
      setStep('confirmation');
    }
  };

  const handleConfirm = () => {
    if (confirmationText === requiredText) {
      onConfirm({
        ...cleanupOptions,
        confirmation_text: confirmationText
      });
      handleClose();
    } else {
      toast.error(`Please type "${requiredText}" exactly to confirm`);
    }
  };

  const getAffectedCount = () => {
    if (!databaseStats) return 0;
    
    let total = 0;
    if (cleanupOptions.clean_papers) total += databaseStats.papers_count || 0;
    if (cleanupOptions.clean_ratings) total += databaseStats.user_ratings_count || 0;
    if (cleanupOptions.clean_embeddings) total += databaseStats.paper_embeddings_count || 0;
    if (cleanupOptions.clean_search_history) total += databaseStats.search_history_count || 0;
    if (cleanupOptions.clean_recommendations) total += databaseStats.recommendations_history_count || 0;
    
    return total;
  };

  const renderWarningStep = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="text-center"
    >
      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-red-100 mb-6">
        <ExclamationTriangleIcon className="h-8 w-8 text-red-600" />
      </div>
      
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        ⚠️ Dangerous Operation Warning
      </h3>
      
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <p className="text-sm text-red-800 font-medium mb-2">
          You are about to permanently delete data from your database:
        </p>
        <ul className="text-sm text-red-700 space-y-1">
          <li>• Research papers and metadata</li>
          <li>• Your personal ratings and notes</li>
          <li>• Expensive AI-generated embeddings</li>
          <li>• Search and recommendation history</li>
        </ul>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <div className="flex items-center gap-2 mb-2">
          <ShieldCheckIcon className="h-5 w-5 text-blue-600" />
          <span className="font-medium text-blue-900">Safety Measures</span>
        </div>
        <p className="text-sm text-blue-800">
          A complete backup will be created automatically before any deletion.
          You can restore your data from this backup if needed.
        </p>
      </div>

      <div className="flex items-center justify-center gap-2 mb-6">
        {countdown > 0 ? (
          <>
            <ClockIcon className="h-5 w-5 text-amber-500" />
            <span className="text-sm text-amber-700">
              Please wait {countdown} seconds to continue...
            </span>
          </>
        ) : (
          <>
            <ShieldCheckIcon className="h-5 w-5 text-green-500" />
            <span className="text-sm text-green-700">
              You may now proceed to the next step
            </span>
          </>
        )}
      </div>

      <div className="flex gap-3">
        <button
          onClick={handleClose}
          className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 border border-transparent rounded-lg hover:bg-gray-200 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleNextStep}
          disabled={!canProceed}
          className="flex-1 px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {canProceed ? 'Continue' : `Wait ${countdown}s`}
        </button>
      </div>
    </motion.div>
  );

  const renderOptionsStep = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
          <DocumentDuplicateIcon className="h-5 w-5 text-blue-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Cleanup Options</h3>
          <p className="text-sm text-gray-600">Choose what to keep and what to delete</p>
        </div>
      </div>

      <div className="space-y-4 mb-6">
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <h4 className="font-medium text-amber-900 mb-3">💡 Recommendations</h4>
          <ul className="text-sm text-amber-800 space-y-1">
            <li>• Keep your ratings - they're valuable for future recommendations</li>
            <li>• Keep embeddings - they're expensive to regenerate (AI costs money)</li>
            <li>• Clean papers only if starting a completely new research topic</li>
          </ul>
        </div>

        {databaseStats && (
          <div className="space-y-3">
            <CleanupOption
              label="Papers & Metadata"
              description="All research papers and their information"
              count={databaseStats.papers_count}
              checked={cleanupOptions.clean_papers}
              onChange={(checked) => setCleanupOptions(prev => ({ ...prev, clean_papers: checked }))}
              danger={true}
            />
            
            <CleanupOption
              label="User Ratings & Notes"
              description="Your personal ratings and notes on papers"
              count={databaseStats.user_ratings_count}
              checked={cleanupOptions.clean_ratings}
              onChange={(checked) => setCleanupOptions(prev => ({ ...prev, clean_ratings: checked }))}
              danger={true}
              recommended={false}
            />
            
            <CleanupOption
              label="AI Embeddings"
              description="Expensive AI-generated paper embeddings (costs money to regenerate)"
              count={databaseStats.paper_embeddings_count}
              checked={cleanupOptions.clean_embeddings}
              onChange={(checked) => setCleanupOptions(prev => ({ ...prev, clean_embeddings: checked }))}
              danger={true}
              recommended={false}
            />
            
            <CleanupOption
              label="Search History"
              description="Your search history and patterns"
              count={databaseStats.search_history_count}
              checked={cleanupOptions.clean_search_history}
              onChange={(checked) => setCleanupOptions(prev => ({ ...prev, clean_search_history: checked }))}
            />
            
            <CleanupOption
              label="Recommendation History"
              description="AI-generated recommendation history"
              count={databaseStats.recommendations_history_count}
              checked={cleanupOptions.clean_recommendations}
              onChange={(checked) => setCleanupOptions(prev => ({ ...prev, clean_recommendations: checked }))}
            />
          </div>
        )}
      </div>

      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <p className="text-sm text-red-800 font-medium">
          {getAffectedCount()} total records will be deleted
        </p>
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => setStep('warning')}
          className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 border border-transparent rounded-lg hover:bg-gray-200 transition-colors"
        >
          Back
        </button>
        <button
          onClick={handleNextStep}
          className="flex-1 px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-lg hover:bg-red-700 transition-colors"
        >
          Continue to Confirmation
        </button>
      </div>
    </motion.div>
  );

  const renderConfirmationStep = () => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
          <TrashIcon className="h-5 w-5 text-red-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Final Confirmation</h3>
          <p className="text-sm text-gray-600">Type the confirmation text to proceed</p>
        </div>
      </div>

      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <p className="text-sm text-red-800 font-medium mb-2">
          You are about to delete {getAffectedCount()} records including:
        </p>
        <ul className="text-sm text-red-700 space-y-1">
          {cleanupOptions.clean_papers && <li>• All research papers</li>}
          {cleanupOptions.clean_ratings && <li>• All your ratings and notes</li>}
          {cleanupOptions.clean_embeddings && <li>• All AI embeddings (expensive to regenerate)</li>}
          {cleanupOptions.clean_search_history && <li>• All search history</li>}
          {cleanupOptions.clean_recommendations && <li>• All recommendation history</li>}
        </ul>
      </div>

      <div className="mb-6">
        <label htmlFor="confirmation" className="block text-sm font-medium text-gray-700 mb-2">
          Type "<strong>{requiredText}</strong>" to confirm deletion:
        </label>
        <input
          type="text"
          id="confirmation"
          value={confirmationText}
          onChange={(e) => setConfirmationText(e.target.value)}
          placeholder={requiredText}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 transition-colors"
        />
        {confirmationText && confirmationText !== requiredText && (
          <p className="mt-1 text-sm text-red-600">
            Text doesn't match. Please type exactly: {requiredText}
          </p>
        )}
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => setStep('options')}
          className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 border border-transparent rounded-lg hover:bg-gray-200 transition-colors"
        >
          Back
        </button>
        <button
          onClick={handleConfirm}
          disabled={confirmationText !== requiredText}
          className="flex-1 px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          <TrashIcon className="h-4 w-4" />
          Delete Data
        </button>
      </div>
    </motion.div>
  );

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={handleClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-25" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <Dialog.Title className="text-lg font-medium text-gray-900">
                    {step === 'warning' && 'Cleanup Warning'}
                    {step === 'options' && 'Cleanup Options'}
                    {step === 'confirmation' && 'Final Confirmation'}
                  </Dialog.Title>
                  <button
                    onClick={handleClose}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <XMarkIcon className="h-5 w-5" />
                  </button>
                </div>

                {/* Progress indicator */}
                <div className="flex items-center gap-2 mb-6">
                  <div className={`w-2 h-2 rounded-full ${step === 'warning' ? 'bg-red-500' : 'bg-gray-300'}`} />
                  <div className={`w-2 h-2 rounded-full ${step === 'options' ? 'bg-red-500' : 'bg-gray-300'}`} />
                  <div className={`w-2 h-2 rounded-full ${step === 'confirmation' ? 'bg-red-500' : 'bg-gray-300'}`} />
                </div>

                {/* Step content */}
                <AnimatePresence mode="wait">
                  {step === 'warning' && renderWarningStep()}
                  {step === 'options' && renderOptionsStep()}
                  {step === 'confirmation' && renderConfirmationStep()}
                </AnimatePresence>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

interface CleanupOptionProps {
  label: string;
  description: string;
  count?: number;
  checked: boolean;
  onChange: (checked: boolean) => void;
  danger?: boolean;
  recommended?: boolean;
}

const CleanupOption: React.FC<CleanupOptionProps> = ({
  label,
  description,
  count,
  checked,
  onChange,
  danger = false,
  recommended = true,
}) => {
  return (
    <div className={`flex items-start space-x-3 p-3 rounded-lg border ${
      checked 
        ? danger 
          ? 'border-red-200 bg-red-50' 
          : 'border-blue-200 bg-blue-50'
        : 'border-gray-200 bg-gray-50'
    }`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-900">
            {label}
          </label>
          {count !== undefined && (
            <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded">
              {count.toLocaleString()} items
            </span>
          )}
          {!recommended && (
            <span className="text-xs text-amber-600 bg-amber-100 px-2 py-1 rounded">
              Not recommended
            </span>
          )}
        </div>
        <p className="text-xs text-gray-600 mt-1">
          {description}
        </p>
      </div>
    </div>
  );
};