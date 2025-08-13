import { apiClient } from './api';

export interface AnalyticsOverview {
  overview: {
    total_papers: number;
    rated_papers: number;
    unrated_papers: number;
    completion_rate: number;
    average_rating: number;
  };
  rating_distribution: Record<number, number>;
  category_breakdown: Record<string, {
    total_papers: number;
    rated_papers: number;
    average_rating: number;
    percentage: number;
  }>;
  summary: {
    most_common_rating: number | null;
    total_categories: number;
    best_rated_category: string | null;
  };
}

class AnalyticsService {
  async getOverview(): Promise<AnalyticsOverview> {
    return apiClient.get<AnalyticsOverview>('/analytics/overview');
  }

  async getPaperStats() {
    return apiClient.get('/papers/stats');
  }
}

export const analyticsService = new AnalyticsService();