import { Middleware } from '@reduxjs/toolkit';
import { setOnlineStatus, addNotification } from '../slices/systemSlice';

export const apiMiddleware: Middleware = (store) => (next) => (action) => {
  // Handle online/offline status
  if (typeof window !== 'undefined') {
    const handleOnline = () => store.dispatch(setOnlineStatus(true));
    const handleOffline = () => store.dispatch(setOnlineStatus(false));
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    // Cleanup function would go here in a real implementation
  }
  
  // Handle API errors globally
  if (action.type.endsWith('/rejected')) {
    const error = action.error;
    
    // Add notification for API errors
    store.dispatch(addNotification({
      type: 'error',
      message: error.message || 'An error occurred',
    }));
    
    // Handle specific error types
    if (error.message?.includes('Network Error') || error.message?.includes('fetch')) {
      store.dispatch(setOnlineStatus(false));
    }
  }
  
  // Handle successful API calls
  if (action.type.endsWith('/fulfilled')) {
    // Update online status on successful API call
    store.dispatch(setOnlineStatus(true));
  }
  
  return next(action);
};