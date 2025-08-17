import React, { useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { 
  XMarkIcon, 
  DocumentPlusIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon 
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';

import { useAppDispatch } from '@store/index';
import { LoadingSpinner } from '@components/common/LoadingSpinner';
import { papersService } from '@services/papersService';

interface ManualPaperModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (paper: any) => void;
}

interface FormData {
  arxivId: string;
  category?: string;
}

interface FormErrors {
  arxivId?: string;
  category?: string;
}

export const ManualPaperModal: React.FC<ManualPaperModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const dispatch = useAppDispatch();
  
  const [formData, setFormData] = useState<FormData>({
    arxivId: '',
    category: '',
  });
  
  const [errors, setErrors] = useState<FormErrors>({});
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<any>(null);

  // Validate ArXiv ID format
  const validateArxivId = (id: string): boolean => {
    const pattern = /^\d{4}\.\d{4,5}(v\d+)?$/;
    return pattern.test(id.trim());
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Clear previous errors and success
    setErrors({});
    setSuccess(null);
    
    // Validate form
    const newErrors: FormErrors = {};
    
    if (!formData.arxivId.trim()) {
      newErrors.arxivId = 'ArXiv ID is required';
    } else if (!validateArxivId(formData.arxivId)) {
      newErrors.arxivId = 'Invalid ArXiv ID format. Expected: YYYY.NNNNN or YYYY.NNNNNvN';
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    setLoading(true);
    
    try {
      const result = await papersService.addManualPaper({
        arxiv_id: formData.arxivId.trim(),
        category: formData.category?.trim() || undefined,
      });
      
      if (result.success) {
        setSuccess(result.data);
        
        if (result.data.already_exists) {
          toast.success(`Paper ${formData.arxivId} already exists in database`);
        } else {
          toast.success(`Successfully added paper ${formData.arxivId}`);
        }
        
        // Call success callback
        if (onSuccess) {
          onSuccess(result.data.paper);
        }
        
        // Reset form after success
        setTimeout(() => {
          setFormData({ arxivId: '', category: '' });
          setSuccess(null);
          onClose();
        }, 2000);
        
      } else {
        throw new Error(result.message || 'Failed to add paper');
      }
      
    } catch (error: any) {
      console.error('Manual paper entry failed:', error);
      
      if (error.response?.status === 400) {
        setErrors({ arxivId: error.response.data.detail || 'Invalid ArXiv ID' });
      } else if (error.response?.status === 404) {
        setErrors({ arxivId: 'Paper not found on ArXiv' });
      } else {
        toast.error(error.message || 'Failed to add paper');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    if (!loading) {
      setFormData({ arxivId: '', category: '' });
      setErrors({});
      setSuccess(null);
      onClose();
    }
  };

  const handleArxivIdChange = (value: string) => {
    setFormData(prev => ({ ...prev, arxivId: value }));
    
    // Clear error when user starts typing
    if (errors.arxivId) {
      setErrors(prev => ({ ...prev, arxivId: undefined }));
    }
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
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2">
                    <DocumentPlusIcon className="h-6 w-6 text-blue-600" />
                    <Dialog.Title
                      as="h3"
                      className="text-lg font-medium leading-6 text-gray-900"
                    >
                      Add Paper Manually
                    </Dialog.Title>
                  </div>
                  <button
                    onClick={handleClose}
                    disabled={loading}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <XMarkIcon className="h-6 w-6" />
                  </button>
                </div>

                {success ? (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-4"
                  >
                    <div className="flex items-center space-x-2 text-green-600">
                      <CheckCircleIcon className="h-6 w-6" />
                      <span className="font-medium">
                        {success.already_exists ? 'Paper Found in Database' : 'Paper Added Successfully'}
                      </span>
                    </div>
                    
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <h4 className="font-medium text-green-800 mb-2">
                        {success.paper.title}
                      </h4>
                      <p className="text-sm text-green-600">
                        <strong>ArXiv ID:</strong> {success.arxiv_id}
                      </p>
                      <p className="text-sm text-green-600">
                        <strong>Category:</strong> {success.paper.category}
                      </p>
                      {success.paper.authors && (
                        <p className="text-sm text-green-600">
                          <strong>Authors:</strong> {
                            Array.isArray(success.paper.authors) 
                              ? success.paper.authors.join(', ')
                              : success.paper.authors
                          }
                        </p>
                      )}
                    </div>
                  </motion.div>
                ) : (
                  <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                      <label htmlFor="arxivId" className="block text-sm font-medium text-gray-700 mb-1">
                        ArXiv ID *
                      </label>
                      <input
                        type="text"
                        id="arxivId"
                        value={formData.arxivId}
                        onChange={(e) => handleArxivIdChange(e.target.value)}
                        placeholder="e.g., 2301.12345 or 2301.12345v1"
                        className={`w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-1 ${
                          errors.arxivId
                            ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                            : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                        }`}
                        disabled={loading}
                      />
                      {errors.arxivId && (
                        <div className="mt-1 flex items-center space-x-1 text-red-600 text-sm">
                          <ExclamationTriangleIcon className="h-4 w-4" />
                          <span>{errors.arxivId}</span>
                        </div>
                      )}
                      <p className="mt-1 text-xs text-gray-500">
                        Format: YYYY.NNNNN (e.g., 2301.12345) with optional version (v1, v2, etc.)
                      </p>
                    </div>

                    <div>
                      <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-1">
                        Category Override (Optional)
                      </label>
                      <input
                        type="text"
                        id="category"
                        value={formData.category}
                        onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value }))}
                        placeholder="e.g., cs.AI, math.OC, quant-ph"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                        disabled={loading}
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        Override the paper's category if needed (uses ArXiv's original category by default)
                      </p>
                    </div>

                    <div className="flex justify-end space-x-3 pt-4">
                      <button
                        type="button"
                        onClick={handleClose}
                        disabled={loading}
                        className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        disabled={loading || !formData.arxivId.trim()}
                        className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 flex items-center space-x-2"
                      >
                        {loading ? (
                          <>
                            <LoadingSpinner size="sm" />
                            <span>Adding Paper...</span>
                          </>
                        ) : (
                          <>
                            <DocumentPlusIcon className="h-4 w-4" />
                            <span>Add Paper</span>
                          </>
                        )}
                      </button>
                    </div>
                  </form>
                )}
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};