import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { CacheStats, SystemConfig } from '../../types';
import { systemService } from '@services/systemService';

interface SystemState {
  config: SystemConfig | null;
  cache: CacheStats | null;
  isOnline: boolean;
  lastSync: string | null;
  loading: {
    config: boolean;
    cache: boolean;
    clearCache: boolean;
    generateRecommendations: boolean;
  };
  error: string | null;
  notifications: Array<{
    id: string;
    type: 'info' | 'warning' | 'error' | 'success';
    message: string;
    timestamp: string;
    read: boolean;
  }>;
}

const initialState: SystemState = {
  config: null,
  cache: null,
  isOnline: navigator.onLine,
  lastSync: null,
  loading: {
    config: false,
    cache: false,
    clearCache: false,
    generateRecommendations: false,
  },
  error: null,
  notifications: [],
};

// Async thunks
export const fetchSystemConfig = createAsyncThunk(
  'system/fetchConfig',
  async () => {
    const response = await systemService.getConfig();
    return response;
  }
);

export const fetchCacheStats = createAsyncThunk(
  'system/fetchCacheStats',
  async () => {
    const response = await systemService.getCacheStats();
    return response;
  }
);

export const clearCache = createAsyncThunk(
  'system/clearCache',
  async () => {
    const response = await systemService.clearCache();
    return response;
  }
);

export const generateRecommendations = createAsyncThunk(
  'system/generateRecommendations',
  async () => {
    const response = await systemService.generateRecommendations();
    return response;
  }
);

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setOnlineStatus: (state, action: PayloadAction<boolean>) => {
      state.isOnline = action.payload;
      
      if (action.payload) {
        // Back online - add notification
        state.notifications.push({
          id: `online-${Date.now()}`,
          type: 'success',
          message: 'Connection restored',
          timestamp: new Date().toISOString(),
          read: false,
        });
      } else {
        // Gone offline - add notification
        state.notifications.push({
          id: `offline-${Date.now()}`,
          type: 'warning',
          message: 'Working offline - changes will sync when connection is restored',
          timestamp: new Date().toISOString(),
          read: false,
        });
      }
    },
    
    updateLastSync: (state) => {
      state.lastSync = new Date().toISOString();
    },
    
    addNotification: (state, action: PayloadAction<Omit<SystemState['notifications'][0], 'id' | 'timestamp' | 'read'>>) => {
      state.notifications.push({
        ...action.payload,
        id: `notification-${Date.now()}`,
        timestamp: new Date().toISOString(),
        read: false,
      });
      
      // Keep only last 50 notifications
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(-50);
      }
    },
    
    markNotificationRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    
    markAllNotificationsRead: (state) => {
      state.notifications.forEach(notification => {
        notification.read = true;
      });
    },
    
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    
    clearAllNotifications: (state) => {
      state.notifications = [];
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    // Budget monitoring
    updateBudgetAlert: (state, action: PayloadAction<{ used: number; limit: number }>) => {
      const { used, limit } = action.payload;
      const percentage = (used / limit) * 100;
      
      if (percentage >= 90) {
        state.notifications.push({
          id: `budget-critical-${Date.now()}`,
          type: 'error',
          message: `Budget usage critical: ${percentage.toFixed(1)}% used`,
          timestamp: new Date().toISOString(),
          read: false,
        });
      } else if (percentage >= 75) {
        state.notifications.push({
          id: `budget-warning-${Date.now()}`,
          type: 'warning',
          message: `Budget usage high: ${percentage.toFixed(1)}% used`,
          timestamp: new Date().toISOString(),
          read: false,
        });
      }
    },
  },
  extraReducers: (builder) => {
    // Fetch system config
    builder
      .addCase(fetchSystemConfig.pending, (state) => {
        state.loading.config = true;
        state.error = null;
      })
      .addCase(fetchSystemConfig.fulfilled, (state, action) => {
        state.loading.config = false;
        state.config = action.payload;
      })
      .addCase(fetchSystemConfig.rejected, (state, action) => {
        state.loading.config = false;
        state.error = action.error.message || 'Failed to fetch system config';
      });

    // Fetch cache stats
    builder
      .addCase(fetchCacheStats.pending, (state) => {
        state.loading.cache = true;
      })
      .addCase(fetchCacheStats.fulfilled, (state, action) => {
        state.loading.cache = false;
        state.cache = action.payload;
      })
      .addCase(fetchCacheStats.rejected, (state, action) => {
        state.loading.cache = false;
        state.error = action.error.message || 'Failed to fetch cache stats';
      });

    // Clear cache
    builder
      .addCase(clearCache.pending, (state) => {
        state.loading.clearCache = true;
      })
      .addCase(clearCache.fulfilled, (state) => {
        state.loading.clearCache = false;
        // Reset cache stats
        if (state.cache) {
          state.cache = {
            ...state.cache,
            total_embeddings: 0,
            cache_size_mb: 0,
            hit_rate: 0,
          };
        }
        state.notifications.push({
          id: `cache-cleared-${Date.now()}`,
          type: 'success',
          message: 'Cache cleared successfully',
          timestamp: new Date().toISOString(),
          read: false,
        });
      })
      .addCase(clearCache.rejected, (state, action) => {
        state.loading.clearCache = false;
        state.error = action.error.message || 'Failed to clear cache';
      });

    // Generate recommendations
    builder
      .addCase(generateRecommendations.pending, (state) => {
        state.loading.generateRecommendations = true;
      })
      .addCase(generateRecommendations.fulfilled, (state) => {
        state.loading.generateRecommendations = false;
        state.notifications.push({
          id: `recommendations-generated-${Date.now()}`,
          type: 'success',
          message: 'New recommendations generated successfully',
          timestamp: new Date().toISOString(),
          read: false,
        });
      })
      .addCase(generateRecommendations.rejected, (state, action) => {
        state.loading.generateRecommendations = false;
        state.error = action.error.message || 'Failed to generate recommendations';
      });
  },
});

export const {
  setOnlineStatus,
  updateLastSync,
  addNotification,
  markNotificationRead,
  markAllNotificationsRead,
  removeNotification,
  clearAllNotifications,
  clearError,
  updateBudgetAlert,
} = systemSlice.actions;

export default systemSlice.reducer;