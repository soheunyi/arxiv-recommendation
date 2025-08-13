export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  papersPerPage: number;
  defaultView: 'grid' | 'list';
  autoSaveRatings: boolean;
  showLatexPreview: boolean;
  preferredCategories: string[];
  notificationSettings: NotificationSettings;
}

export interface NotificationSettings {
  newRecommendations: boolean;
  weeklyDigest: boolean;
  paperUpdates: boolean;
  systemAlerts: boolean;
}

export interface UserStats {
  total_ratings: number;
  papers_read: number;
  avg_rating: number;
  favorite_categories: string[];
  daily_activity: Array<{
    date: string;
    ratings: number;
    papers_viewed: number;
  }>;
  joined_date: string;
}

export interface UserSession {
  isAuthenticated: boolean;
  preferences: UserPreferences;
  stats: UserStats;
  lastActivity: string;
}