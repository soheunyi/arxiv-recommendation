export type { Paper, PaperFilters, PaperSearchParams, PaperSearchResult } from './paper';
export type { Rating, RatingUpdate, RatingStats, CategoryRating, RatingTimeline } from './rating';
export type { 
  ApiResponse, 
  ApiError, 
  PaginatedResponse, 
  CacheStats, 
  SystemConfig, 
  RecommendationResult 
} from './api';
export type { 
  UserPreferences, 
  NotificationSettings, 
  UserStats, 
  UserSession 
} from './user';

// Common UI types
export interface LoadingState {
  isLoading: boolean;
  error: string | null;
}

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  description?: string;
  duration?: number;
}