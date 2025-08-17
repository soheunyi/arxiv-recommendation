import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { UserPreferences, UserStats, UserSession } from '../../types';

interface UserState {
  session: UserSession;
  loading: {
    preferences: boolean;
    stats: boolean;
  };
  error: string | null;
}

const defaultPreferences: UserPreferences = {
  theme: 'system',
  papersPerPage: 20,
  defaultView: 'list',
  autoSaveRatings: true,
  showLatexPreview: true,
  preferredCategories: [],
  notificationSettings: {
    newRecommendations: true,
    weeklyDigest: false,
    paperUpdates: true,
    systemAlerts: true,
  },
};

const defaultStats: UserStats = {
  total_ratings: 0,
  papers_read: 0,
  avg_rating: 0,
  favorite_categories: [],
  daily_activity: [],
  joined_date: new Date().toISOString(),
};

const initialState: UserState = {
  session: {
    isAuthenticated: true, // For now, assuming single user
    preferences: defaultPreferences,
    stats: defaultStats,
    lastActivity: new Date().toISOString(),
  },
  loading: {
    preferences: false,
    stats: false,
  },
  error: null,
};

const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    updatePreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
      state.session.preferences = {
        ...state.session.preferences,
        ...action.payload,
      };
      state.session.lastActivity = new Date().toISOString();
    },
    
    updateStats: (state, action: PayloadAction<Partial<UserStats>>) => {
      state.session.stats = {
        ...state.session.stats,
        ...action.payload,
      };
    },
    
    incrementRatingCount: (state) => {
      state.session.stats.total_ratings += 1;
      state.session.lastActivity = new Date().toISOString();
    },
    
    updateAverageRating: (state, action: PayloadAction<number>) => {
      state.session.stats.avg_rating = action.payload;
    },
    
    addDailyActivity: (state, action: PayloadAction<{ ratings: number; papers_viewed: number }>) => {
      const today = new Date().toISOString().split('T')[0];
      const existingIndex = state.session.stats.daily_activity.findIndex(
        activity => activity.date === today
      );
      
      const newActivity = {
        date: today,
        ratings: action.payload.ratings,
        papers_viewed: action.payload.papers_viewed,
      };
      
      if (existingIndex >= 0) {
        state.session.stats.daily_activity[existingIndex] = newActivity;
      } else {
        state.session.stats.daily_activity.push(newActivity);
        
        // Keep only last 30 days
        if (state.session.stats.daily_activity.length > 30) {
          state.session.stats.daily_activity = state.session.stats.daily_activity
            .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
            .slice(0, 30);
        }
      }
    },
    
    updateFavoriteCategories: (state, action: PayloadAction<string[]>) => {
      state.session.stats.favorite_categories = action.payload;
    },
    
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'system'>) => {
      state.session.preferences.theme = action.payload;
    },
    
    setDefaultView: (state, action: PayloadAction<'grid' | 'list'>) => {
      state.session.preferences.defaultView = action.payload;
    },
    
    toggleAutoSaveRatings: (state) => {
      state.session.preferences.autoSaveRatings = !state.session.preferences.autoSaveRatings;
    },
    
    toggleLatexPreview: (state) => {
      state.session.preferences.showLatexPreview = !state.session.preferences.showLatexPreview;
    },
    
    updateNotificationSettings: (state, action: PayloadAction<Partial<UserPreferences['notificationSettings']>>) => {
      state.session.preferences.notificationSettings = {
        ...state.session.preferences.notificationSettings,
        ...action.payload,
      };
    },
    
    updateLastActivity: (state) => {
      state.session.lastActivity = new Date().toISOString();
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    // Reset to defaults
    resetPreferences: (state) => {
      state.session.preferences = defaultPreferences;
    },
    
    resetStats: (state) => {
      state.session.stats = defaultStats;
    },
  },
});

export const {
  updatePreferences,
  updateStats,
  incrementRatingCount,
  updateAverageRating,
  addDailyActivity,
  updateFavoriteCategories,
  setTheme,
  setDefaultView,
  toggleAutoSaveRatings,
  toggleLatexPreview,
  updateNotificationSettings,
  updateLastActivity,
  clearError,
  resetPreferences,
  resetStats,
} = userSlice.actions;

export default userSlice.reducer;