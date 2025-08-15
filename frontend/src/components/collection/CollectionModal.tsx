import React, { useState, useEffect } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { 
  XMarkIcon, 
  MagnifyingGlassIcon,
  DocumentPlusIcon,
  ExclamationTriangleIcon,
  CalendarIcon,
  SparklesIcon 
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store';
import { startCollection, closeCollectionModal, fetchLLMProviders } from '@store/slices/papersSlice';
import { LoadingSpinner } from '@components/common/LoadingSpinner';

interface CollectionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CollectionModal: React.FC<CollectionModalProps> = ({
  isOpen,
  onClose,
}) => {
  const dispatch = useAppDispatch();
  const { loading, llmProviders } = useAppSelector((state) => state.papers);
  
  const [formData, setFormData] = useState({
    keyword: '',
    maxPapers: 20,
    cleanDb: false,
    llmProvider: '',
    dateFrom: '',
    dateTo: '',
    useCollaboration: false,
    collaborationStrategy: 'adaptive',
  });
  
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Load LLM providers when modal opens
  useEffect(() => {
    if (isOpen && !llmProviders) {
      dispatch(fetchLLMProviders());
    }
  }, [isOpen, llmProviders, dispatch]);

  // Set default provider when providers are loaded
  useEffect(() => {
    if (llmProviders && !formData.llmProvider) {
      setFormData(prev => ({
        ...prev,
        llmProvider: llmProviders.current
      }));
    }
  }, [llmProviders, formData.llmProvider]);

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

    // Validate date range if provided
    if (formData.dateFrom && formData.dateTo) {
      const fromDate = new Date(formData.dateFrom);
      const toDate = new Date(formData.dateTo);
      
      if (fromDate > toDate) {
        newErrors.dateRange = 'Start date must be before end date';
      }
      
      // Check if dates are reasonable (not too far in the past or future)
      const minDate = new Date('2007-01-01'); // ArXiv started in 1991, but most papers from 2007+
      const maxDate = new Date();
      maxDate.setDate(maxDate.getDate() + 30); // Allow 30 days in the future
      
      if (fromDate < minDate) {
        newErrors.dateFrom = 'Start date should be 2007 or later';
      }
      if (toDate > maxDate) {
        newErrors.dateTo = 'End date cannot be more than 30 days in the future';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      await dispatch(startCollection({
        keyword: formData.keyword.trim(),
        max_papers: formData.maxPapers,
        clean_db: formData.cleanDb,
        llm_provider: formData.llmProvider,
        date_from: formData.dateFrom || undefined,
        date_to: formData.dateTo || undefined,
        use_collaboration: formData.useCollaboration,
        collaboration_strategy: formData.useCollaboration ? formData.collaborationStrategy : undefined,
      })).unwrap();
      
      toast.success('Collection started! This will run in the background.');
      onClose();
      
      // Reset form
      setFormData({
        keyword: '',
        maxPapers: 20,
        cleanDb: false,
        llmProvider: llmProviders?.current || '',
        dateFrom: '',
        dateTo: '',
        useCollaboration: false,
        collaborationStrategy: 'adaptive',
      });
      setErrors({});
      
    } catch (error: any) {
      toast.error(error.message || 'Failed to start collection');
    }
  };

  const handleClose = () => {
    onClose();
    setErrors({});
  };

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
                  <Dialog.Title className="text-lg font-medium text-secondary-900 flex items-center gap-2">
                    <DocumentPlusIcon className="h-5 w-5 text-primary-600" />
                    Collect ArXiv Papers
                  </Dialog.Title>
                  <button
                    onClick={handleClose}
                    className="text-secondary-400 hover:text-secondary-600 transition-colors"
                  >
                    <XMarkIcon className="h-5 w-5" />
                  </button>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-4">
                  {/* Keyword Input */}
                  <div>
                    <label htmlFor="keyword" className="block text-sm font-medium text-secondary-700 mb-1">
                      Research Keyword
                    </label>
                    <div className="relative">
                      <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-secondary-400" />
                      <input
                        type="text"
                        id="keyword"
                        value={formData.keyword}
                        onChange={(e) => setFormData({ ...formData, keyword: e.target.value })}
                        placeholder="e.g., machine learning, optimal transport, quantum computing"
                        className={`w-full pl-10 pr-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors ${
                          errors.keyword ? 'border-red-300' : 'border-secondary-300'
                        }`}
                      />
                    </div>
                    {errors.keyword && (
                      <p className="mt-1 text-sm text-red-600">{errors.keyword}</p>
                    )}
                    <p className="mt-1 text-xs text-secondary-500">
                      AI will generate multiple search queries related to this keyword
                    </p>
                  </div>

                  {/* Max Papers */}
                  <div>
                    <label htmlFor="maxPapers" className="block text-sm font-medium text-secondary-700 mb-1">
                      Maximum Papers
                    </label>
                    <input
                      type="number"
                      id="maxPapers"
                      min="1"
                      max="100"
                      value={formData.maxPapers}
                      onChange={(e) => setFormData({ ...formData, maxPapers: parseInt(e.target.value) || 20 })}
                      className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors ${
                        errors.maxPapers ? 'border-red-300' : 'border-secondary-300'
                      }`}
                    />
                    {errors.maxPapers && (
                      <p className="mt-1 text-sm text-red-600">{errors.maxPapers}</p>
                    )}
                  </div>

                  {/* LLM Provider Selection */}
                  <div>
                    <label htmlFor="llmProvider" className="block text-sm font-medium text-secondary-700 mb-1">
                      LLM Provider
                    </label>
                    {loading.providers ? (
                      <div className="flex items-center space-x-2">
                        <LoadingSpinner size="sm" />
                        <span className="text-sm text-secondary-500">Loading providers...</span>
                      </div>
                    ) : llmProviders ? (
                      <>
                        <select
                          id="llmProvider"
                          value={formData.llmProvider}
                          onChange={(e) => setFormData({ ...formData, llmProvider: e.target.value })}
                          className="w-full px-3 py-2 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
                        >
                          {Object.entries(llmProviders.available).filter(([key]) => key !== 'cost_savings').map(([key, provider]: [string, any]) => (
                            <option key={key} value={key}>
                              {provider.name} ({provider.query_model}) - {key === 'gemini' ? '33x cheaper' : 'premium'}
                            </option>
                          ))}
                        </select>
                        <p className="mt-1 text-xs text-secondary-500">
                          {formData.llmProvider === llmProviders.recommendation && (
                            <span className="text-green-600 font-medium">✨ Recommended - </span>
                          )}
                          Choose the AI model for generating search queries
                        </p>
                        {formData.llmProvider === 'gemini' && (
                          <motion.div 
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            className="mt-2 p-3 bg-green-50 border border-green-200 rounded-lg"
                          >
                            <p className="text-xs text-green-700">
                              <strong>💰 Cost Savings:</strong> Gemini is 33x cheaper than GPT-4o for query generation while providing excellent results.
                            </p>
                          </motion.div>
                        )}
                      </>
                    ) : (
                      <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <p className="text-sm text-yellow-700">
                          Unable to load LLM providers. Using default settings.
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Date Range Filter */}
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <CalendarIcon className="h-4 w-4 text-secondary-500" />
                      <label className="text-sm font-medium text-secondary-700">
                        Date Range Filter (Optional)
                      </label>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label htmlFor="dateFrom" className="block text-xs font-medium text-secondary-600 mb-1">
                          From Date
                        </label>
                        <input
                          type="date"
                          id="dateFrom"
                          value={formData.dateFrom}
                          onChange={(e) => setFormData({ ...formData, dateFrom: e.target.value })}
                          className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors ${
                            errors.dateFrom ? 'border-red-300' : 'border-secondary-300'
                          }`}
                          min="2007-01-01"
                          max={new Date().toISOString().split('T')[0]}
                        />
                        {errors.dateFrom && (
                          <p className="mt-1 text-xs text-red-600">{errors.dateFrom}</p>
                        )}
                      </div>
                      
                      <div>
                        <label htmlFor="dateTo" className="block text-xs font-medium text-secondary-600 mb-1">
                          To Date
                        </label>
                        <input
                          type="date"
                          id="dateTo"
                          value={formData.dateTo}
                          onChange={(e) => setFormData({ ...formData, dateTo: e.target.value })}
                          className={`w-full px-3 py-2 text-sm border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors ${
                            errors.dateTo ? 'border-red-300' : 'border-secondary-300'
                          }`}
                          min="2007-01-01"
                          max={new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]}
                        />
                        {errors.dateTo && (
                          <p className="mt-1 text-xs text-red-600">{errors.dateTo}</p>
                        )}
                      </div>
                    </div>
                    
                    {errors.dateRange && (
                      <p className="text-xs text-red-600">{errors.dateRange}</p>
                    )}
                    
                    <p className="text-xs text-secondary-500">
                      Filter papers by their submission date to ArXiv. Leave empty to search all dates.
                    </p>
                  </div>

                  {/* OpenAI-Gemini Collaboration */}
                  <div className="space-y-3">
                    <div className="flex items-start space-x-3">
                      <input
                        type="checkbox"
                        id="useCollaboration"
                        checked={formData.useCollaboration}
                        onChange={(e) => setFormData({ ...formData, useCollaboration: e.target.checked })}
                        className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                      />
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <SparklesIcon className="h-4 w-4 text-purple-500" />
                          <label htmlFor="useCollaboration" className="text-sm font-medium text-secondary-700">
                            Enable AI Collaboration
                          </label>
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                            Advanced
                          </span>
                        </div>
                        <p className="text-xs text-secondary-500 mt-1">
                          Use both OpenAI and Gemini together for optimal query generation with cost efficiency
                        </p>
                      </div>
                    </div>

                    {formData.useCollaboration && (
                      <motion.div 
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="ml-7 space-y-3"
                      >
                        <div>
                          <label htmlFor="collaborationStrategy" className="block text-xs font-medium text-secondary-600 mb-1">
                            Collaboration Strategy
                          </label>
                          <select
                            id="collaborationStrategy"
                            value={formData.collaborationStrategy}
                            onChange={(e) => setFormData({ ...formData, collaborationStrategy: e.target.value })}
                            className="w-full px-3 py-2 text-sm border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
                          >
                            <option value="adaptive">Adaptive (Recommended) - Auto-select best approach</option>
                            <option value="cost_optimized">Cost Optimized - Gemini primary, OpenAI validation</option>
                            <option value="quality_first">Quality First - OpenAI primary, Gemini backup</option>
                            <option value="parallel_compare">Parallel Compare - Both providers, best result</option>
                          </select>
                        </div>
                        
                        <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
                          <div className="flex items-start space-x-2">
                            <SparklesIcon className="h-4 w-4 text-purple-500 mt-0.5 flex-shrink-0" />
                            <div>
                              <p className="text-xs text-purple-700 font-medium">How Collaboration Works:</p>
                              <ul className="text-xs text-purple-600 mt-1 space-y-1">
                                <li>• <strong>Adaptive:</strong> Automatically selects the best strategy based on topic complexity and budget</li>
                                <li>• <strong>Cost Optimized:</strong> Uses Gemini (33x cheaper) with OpenAI quality validation when needed</li>
                                <li>• <strong>Quality First:</strong> Uses OpenAI for best results with Gemini as backup</li>
                                <li>• <strong>Parallel Compare:</strong> Runs both providers and selects the highest quality result</li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </div>

                  {/* Clean Database Option */}
                  <div className="flex items-start space-x-3">
                    <input
                      type="checkbox"
                      id="cleanDb"
                      checked={formData.cleanDb}
                      onChange={(e) => setFormData({ ...formData, cleanDb: e.target.checked })}
                      className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <div>
                      <label htmlFor="cleanDb" className="text-sm font-medium text-secondary-700">
                        Clean existing papers
                      </label>
                      {formData.cleanDb && (
                        <motion.div 
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="mt-2 p-3 bg-orange-50 border border-orange-200 rounded-lg"
                        >
                          <div className="flex items-start space-x-2">
                            <ExclamationTriangleIcon className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-orange-700">
                              This will remove all existing papers and ratings before collecting new ones. This action cannot be undone.
                            </p>
                          </div>
                        </motion.div>
                      )}
                      <p className="text-xs text-secondary-500 mt-1">
                        Remove all current papers before collecting new ones
                      </p>
                    </div>
                  </div>

                  {/* Info Box */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <p className="text-sm text-blue-700">
                      <strong>How it works:</strong> The AI will generate smart search queries related to your keyword
                      {formData.dateFrom || formData.dateTo ? ', filtered by your specified date range,' : ''} 
                      then fetch relevant papers from ArXiv.
                      {formData.useCollaboration ? ' With collaboration enabled, multiple AI providers work together for optimal results.' : ''}
                      {' '}This process runs in the background and typically takes 1-3 minutes.
                    </p>
                    {(formData.dateFrom || formData.dateTo) && (
                      <p className="text-xs text-blue-600 mt-2">
                        <strong>📅 Date filtering:</strong> Only papers submitted to ArXiv within your date range will be collected.
                      </p>
                    )}
                    {formData.useCollaboration && (
                      <p className="text-xs text-blue-600 mt-2">
                        <strong>🤖 AI Collaboration:</strong> Using the {formData.collaborationStrategy} strategy for enhanced query generation.
                      </p>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex space-x-3 pt-4">
                    <button
                      type="button"
                      onClick={handleClose}
                      className="flex-1 px-4 py-2 text-sm font-medium text-secondary-700 bg-secondary-100 border border-transparent rounded-lg hover:bg-secondary-200 focus:outline-none focus:ring-2 focus:ring-secondary-500 transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={loading.collection}
                      className="flex-1 px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-lg hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                    >
                      {loading.collection ? (
                        <>
                          <LoadingSpinner size="sm" />
                          Starting...
                        </>
                      ) : (
                        <>
                          <DocumentPlusIcon className="h-4 w-4" />
                          Start Collection
                        </>
                      )}
                    </button>
                  </div>
                </form>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};