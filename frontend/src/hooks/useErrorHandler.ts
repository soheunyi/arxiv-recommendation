import { useState, useEffect, useCallback } from 'react';
import { ApiError } from '../types/api';

interface ErrorState {
  currentError: ApiError | null;
  errorHistory: ApiError[];
}

export const useErrorHandler = () => {
  const [errorState, setErrorState] = useState<ErrorState>({
    currentError: null,
    errorHistory: [],
  });

  // Listen for global API errors
  useEffect(() => {
    const handleApiError = (event: CustomEvent<ApiError>) => {
      const error = event.detail;
      
      setErrorState(prev => ({
        currentError: error,
        errorHistory: [error, ...prev.errorHistory.slice(0, 9)], // Keep last 10 errors
      }));

      // Auto-dismiss after 10 seconds for non-critical errors
      if (error.statusCode < 500) {
        setTimeout(() => {
          setErrorState(prev => ({
            ...prev,
            currentError: prev.currentError?.correlationId === error.correlationId ? null : prev.currentError,
          }));
        }, 10000);
      }
    };

    window.addEventListener('api-error', handleApiError as EventListener);

    return () => {
      window.removeEventListener('api-error', handleApiError as EventListener);
    };
  }, []);

  const dismissError = useCallback(() => {
    setErrorState(prev => ({
      ...prev,
      currentError: null,
    }));
  }, []);

  const clearErrorHistory = useCallback(() => {
    setErrorState(prev => ({
      ...prev,
      errorHistory: [],
    }));
  }, []);

  const showError = useCallback((error: ApiError) => {
    setErrorState(prev => ({
      currentError: error,
      errorHistory: [error, ...prev.errorHistory.slice(0, 9)],
    }));
  }, []);

  return {
    currentError: errorState.currentError,
    errorHistory: errorState.errorHistory,
    dismissError,
    clearErrorHistory,
    showError,
  };
};

export default useErrorHandler;