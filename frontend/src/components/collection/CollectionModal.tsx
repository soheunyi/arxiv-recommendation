import React, { useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { 
  XMarkIcon, 
  MagnifyingGlassIcon,
  DocumentPlusIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';

import { useAppDispatch, useAppSelector } from '@store';
import { startCollection, closeCollectionModal } from '@store/slices/papersSlice';
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
  const { loading } = useAppSelector((state) => state.papers);
  
  const [formData, setFormData] = useState({
    keyword: '',
    maxPapers: 20,
    cleanDb: false,
  });
  
  const [errors, setErrors] = useState<Record<string, string>>({});

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
      })).unwrap();
      
      toast.success('Collection started! This will run in the background.');
      onClose();
      
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
                      GPT will generate multiple search queries related to this keyword
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
                      <strong>How it works:</strong> GPT will generate smart search queries related to your keyword, 
                      then fetch relevant papers from ArXiv. This process runs in the background and typically takes 1-2 minutes.
                    </p>
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