import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  DocumentPlusIcon,
  MagnifyingGlassIcon,
  SparklesIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  BookOpenIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store/index';
import { startCollection } from '@store/slices/papersSlice';
import { LoadingSpinner } from '@components/common/LoadingSpinner';
import { CollectionProgress } from '@components/collection/CollectionProgress';
import { DatabaseCleanupModal } from '@components/collection/DatabaseCleanupModal';

export const CollectPapersPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { loading } = useAppSelector((state) => state.papers);
  
  const [formData, setFormData] = useState({
    keyword: '',
    maxPapers: 20,
    cleanDb: false,
  });
  
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showCleanupModal, setShowCleanupModal] = useState(false);
  const [databaseStats, setDatabaseStats] = useState<any>(null);

  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.keyword.trim()) {
      newErrors.keyword = 'Research keyword is required';
    } else if (formData.keyword.trim().length < 3) {
      newErrors.keyword = 'Keyword must be at least 3 characters';
    }
    
    if (formData.maxPapers < 1 || formData.maxPapers > 100) {
      newErrors.maxPapers = 'Max papers must be between 1 and 100';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Load database stats on component mount
  useEffect(() => {
    const loadDatabaseStats = async () => {
      try {
        const response = await fetch('/api/papers/stats');
        if (response.ok) {
          const data = await response.json();
          setDatabaseStats(data.data);
        }
      } catch (error) {
        console.error('Failed to load database stats:', error);
      }
    };

    loadDatabaseStats();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    // If clean database is requested, show the enhanced modal
    if (formData.cleanDb) {
      setShowCleanupModal(true);
      return;
    }

    // Otherwise proceed normally
    await executeCollection();
  };

  const executeCollection = async (cleanupOptions?: any) => {
    try {
      await dispatch(startCollection({
        keyword: formData.keyword.trim(),
        max_papers: formData.maxPapers,
        clean_db: formData.cleanDb,
        cleanup_options: cleanupOptions,
      })).unwrap();
      
      toast.success('Collection started! This will run in the background.');
      
      // Reset form
      setFormData({
        keyword: '',
        maxPapers: 20,
        cleanDb: false,
      });
      setErrors({});
      
    } catch (error: any) {
      toast.error(error.message || 'Failed to start collection');
    }
  };

  const handleCleanupConfirm = async (cleanupOptions: any) => {
    setShowCleanupModal(false);
    await executeCollection(cleanupOptions);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-secondary-900 flex items-center gap-3">
          <DocumentPlusIcon className="h-8 w-8 text-primary-600" />
          Collect Papers
        </h1>
        <p className="mt-2 text-secondary-600">
          Discover and collect relevant research papers from arXiv based on your research interests
        </p>
      </div>

      {/* Collection Progress */}
      <CollectionProgress />

      {/* Main Collection Form */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Collection Form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white rounded-xl border border-secondary-200 shadow-sm p-6"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
              <SparklesIcon className="h-5 w-5 text-primary-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-secondary-900">Start Collection</h2>
              <p className="text-sm text-secondary-600">Configure your paper collection parameters</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Keyword Input */}
            <div>
              <label htmlFor="keyword" className="block text-sm font-medium text-secondary-700 mb-2">
                Research Keyword
              </label>
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-secondary-400" />
                <input
                  type="text"
                  id="keyword"
                  value={formData.keyword}
                  onChange={(e) => setFormData({ ...formData, keyword: e.target.value })}
                  placeholder="e.g., machine learning, optimal transport, quantum computing"
                  className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors ${
                    errors.keyword ? 'border-red-300' : 'border-secondary-300'
                  }`}
                />
              </div>
              {errors.keyword && (
                <p className="mt-1 text-sm text-red-600">{errors.keyword}</p>
              )}
              <p className="mt-2 text-sm text-secondary-500">
                GPT will generate multiple smart search queries related to this keyword
              </p>
            </div>

            {/* Max Papers */}
            <div>
              <label htmlFor="maxPapers" className="block text-sm font-medium text-secondary-700 mb-2">
                Maximum Papers
              </label>
              <input
                type="number"
                id="maxPapers"
                min="1"
                max="100"
                value={formData.maxPapers}
                onChange={(e) => setFormData({ ...formData, maxPapers: parseInt(e.target.value) || 20 })}
                className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors ${
                  errors.maxPapers ? 'border-red-300' : 'border-secondary-300'
                }`}
              />
              {errors.maxPapers && (
                <p className="mt-1 text-sm text-red-600">{errors.maxPapers}</p>
              )}
              <p className="mt-2 text-sm text-secondary-500">
                Number of papers to collect (1-100)
              </p>
            </div>

            {/* Enhanced Clean Database Option */}
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="cleanDb"
                  checked={formData.cleanDb}
                  onChange={(e) => setFormData({ ...formData, cleanDb: e.target.checked })}
                  className="mt-1 h-4 w-4 text-red-600 focus:ring-red-500 border-red-300 rounded"
                />
                <div className="flex-1">
                  <label htmlFor="cleanDb" className="text-sm font-medium text-red-800 flex items-center gap-2">
                    <ExclamationTriangleIcon className="h-4 w-4 text-red-600" />
                    Clean existing database (Dangerous)
                  </label>
                  
                  {formData.cleanDb && (
                    <div className="mt-3 p-3 bg-red-100 border border-red-300 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <ShieldCheckIcon className="h-4 w-4 text-blue-600" />
                        <span className="text-sm font-medium text-red-900">Enhanced Protection Active</span>
                      </div>
                      <ul className="text-xs text-red-800 space-y-1">
                        <li>• Automatic backup will be created before deletion</li>
                        <li>• Multi-step confirmation with safety countdown</li>
                        <li>• Granular options to keep ratings and embeddings</li>
                        <li>• Full recovery options from backup</li>
                      </ul>
                    </div>
                  )}
                  
                  <p className="text-sm text-red-700 mt-2">
                    This will open a protected deletion wizard with backup creation and granular options.
                  </p>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <motion.button
              type="submit"
              disabled={loading.collection}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full px-6 py-3 text-white bg-primary-600 border border-transparent rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-3 font-medium"
            >
              {loading.collection ? (
                <>
                  <LoadingSpinner size="sm" />
                  Starting Collection...
                </>
              ) : (
                <>
                  <ArrowPathIcon className="h-5 w-5" />
                  Start Collection
                </>
              )}
            </motion.button>
          </form>
        </motion.div>

        {/* Information Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          {/* How it Works */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                <InformationCircleIcon className="h-5 w-5 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-blue-900">How it Works</h3>
            </div>
            <div className="space-y-3 text-sm text-blue-800">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-200 rounded-full flex items-center justify-center text-blue-800 font-bold text-xs flex-shrink-0 mt-0.5">1</div>
                <p>GPT generates smart search queries related to your keyword</p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-200 rounded-full flex items-center justify-center text-blue-800 font-bold text-xs flex-shrink-0 mt-0.5">2</div>
                <p>System searches arXiv for relevant papers using those queries</p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-200 rounded-full flex items-center justify-center text-blue-800 font-bold text-xs flex-shrink-0 mt-0.5">3</div>
                <p>Papers are processed and added to your collection with embeddings</p>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-200 rounded-full flex items-center justify-center text-blue-800 font-bold text-xs flex-shrink-0 mt-0.5">4</div>
                <p>You can rate and explore the collected papers</p>
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="bg-green-50 border border-green-200 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                <BookOpenIcon className="h-5 w-5 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-green-900">Collection Tips</h3>
            </div>
            <ul className="space-y-2 text-sm text-green-800">
              <li className="flex items-start gap-2">
                <span className="text-green-600 font-bold">•</span>
                <span>Use specific research topics for better results (e.g., "neural networks" instead of "AI")</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 font-bold">•</span>
                <span>Start with 20-50 papers to get a good overview of the field</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 font-bold">•</span>
                <span>Collection typically takes 1-2 minutes and runs in the background</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 font-bold">•</span>
                <span>You can rate papers to improve future recommendations</span>
              </li>
            </ul>
          </div>

          {/* Quick Examples */}
          <div className="bg-secondary-50 border border-secondary-200 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-secondary-900 mb-4">Example Keywords</h3>
            <div className="grid grid-cols-1 gap-2">
              {[
                'machine learning',
                'optimal transport',
                'quantum computing',
                'computer vision',
                'natural language processing',
                'reinforcement learning'
              ].map((keyword) => (
                <button
                  key={keyword}
                  onClick={() => setFormData({ ...formData, keyword })}
                  className="text-left px-3 py-2 bg-white border border-secondary-200 rounded-lg hover:bg-secondary-50 hover:border-primary-300 transition-colors text-sm"
                >
                  {keyword}
                </button>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Enhanced Database Cleanup Modal */}
      <DatabaseCleanupModal
        isOpen={showCleanupModal}
        onClose={() => setShowCleanupModal(false)}
        onConfirm={handleCleanupConfirm}
        databaseStats={databaseStats}
      />
    </div>
  );
};