export { apiClient, ApiClient } from './api';
export { papersService, PapersService } from './papersService';
export { ratingsService, RatingsService } from './ratingsService';
export { systemService, SystemService } from './systemService';

// Analytics service
export class AnalyticsService {
  // Combined analytics from multiple services
  async getDashboardAnalytics() {
    const [ratingStats, cacheStats, paperStats] = await Promise.all([
      ratingsService.getRatingStats(),
      systemService.getCacheStats(),
      papersService.getPaperStats(),
    ]);
    
    return {
      ratings: ratingStats,
      cache: cacheStats,
      papers: paperStats,
      generated_at: new Date().toISOString(),
    };
  }
  
  // Performance analytics
  async getPerformanceAnalytics() {
    const [metrics, apiUsage, health] = await Promise.all([
      systemService.getMetrics(),
      systemService.getApiUsage(),
      systemService.getHealth(),
    ]);
    
    return {
      metrics,
      apiUsage,
      health,
      generated_at: new Date().toISOString(),
    };
  }
}

export const analyticsService = new AnalyticsService();