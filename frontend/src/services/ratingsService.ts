import { apiClient } from './api';
import { Rating, RatingUpdate, RatingStats, CategoryRating, RatingTimeline } from '@types';

export class RatingsService {
  // Get all user ratings
  async getUserRatings(): Promise<Rating[]> {
    return apiClient.get<Rating[]>('/ratings/user');
  }
  
  // Update a single rating
  async updateRating(ratingUpdate: RatingUpdate): Promise<Rating> {
    return apiClient.post<Rating>('/ratings/update', ratingUpdate);
  }
  
  // Batch update multiple ratings
  async batchUpdateRatings(updates: RatingUpdate[]): Promise<Rating[]> {
    return apiClient.post<Rating[]>('/ratings/batch-update', { updates });
  }
  
  // Get rating for specific paper
  async getPaperRating(paperId: string): Promise<Rating | null> {
    try {
      return await apiClient.get<Rating>(`/ratings/paper/${paperId}`);
    } catch (error: any) {
      if (error.statusCode === 404) {
        return null;
      }
      throw error;
    }
  }
  
  // Delete rating
  async deleteRating(paperId: string): Promise<void> {
    return apiClient.delete(`/ratings/paper/${paperId}`);
  }
  
  // Get rating statistics
  async getRatingStats(): Promise<RatingStats> {
    return apiClient.get<RatingStats>('/ratings/stats');
  }
  
  // Get rating distribution
  async getRatingDistribution(): Promise<Record<number, number>> {
    return apiClient.get<Record<number, number>>('/ratings/distribution');
  }
  
  // Get category ratings
  async getCategoryRatings(): Promise<CategoryRating[]> {
    return apiClient.get<CategoryRating[]>('/ratings/categories');
  }
  
  // Get rating timeline
  async getRatingTimeline(days: number = 30): Promise<RatingTimeline[]> {
    return apiClient.get<RatingTimeline[]>(`/ratings/timeline?days=${days}`);
  }
  
  // Get top rated papers
  async getTopRatedPapers(limit: number = 10): Promise<Array<{
    paper_id: string;
    title: string;
    rating: number;
    notes?: string;
  }>> {
    return apiClient.get(`/ratings/top-rated?limit=${limit}`);
  }
  
  // Get recent ratings
  async getRecentRatings(limit: number = 20): Promise<Array<{
    paper_id: string;
    title: string;
    rating: number;
    created_at: string;
  }>> {
    return apiClient.get(`/ratings/recent?limit=${limit}`);
  }
  
  // Get ratings by category
  async getRatingsByCategory(category: string): Promise<Rating[]> {
    return apiClient.get<Rating[]>(`/ratings/category/${encodeURIComponent(category)}`);
  }
  
  // Get unrated papers
  async getUnratedPapers(limit: number = 50): Promise<Array<{
    id: string;
    title: string;
    category: string;
    published_date: string;
  }>> {
    return apiClient.get(`/ratings/unrated?limit=${limit}`);
  }
  
  // Export ratings
  async exportRatings(format: 'json' | 'csv' = 'json'): Promise<Blob> {
    const response = await apiClient.getClient().get(`/ratings/export?format=${format}`, {
      responseType: 'blob',
    });
    
    return response.data;
  }
  
  // Import ratings
  async importRatings(file: File, onProgress?: (progress: number) => void): Promise<{
    imported: number;
    updated: number;
    skipped: number;
    errors: string[];
  }> {
    return apiClient.upload('/ratings/import', file, onProgress);
  }
  
  // Analytics methods
  async getRatingAnalytics(): Promise<{
    total_ratings: number;
    average_rating: number;
    median_rating: number;
    rating_velocity: number; // ratings per day
    category_breakdown: Record<string, {
      count: number;
      average: number;
      percentage: number;
    }>;
    monthly_trends: Array<{
      month: string;
      count: number;
      average: number;
    }>;
  }> {
    return apiClient.get('/ratings/analytics');
  }
  
  // Get rating recommendations (papers similar to highly rated ones)
  async getRatingBasedRecommendations(limit: number = 10): Promise<Array<{
    paper_id: string;
    title: string;
    predicted_rating: number;
    confidence: number;
    similar_papers: string[];
  }>> {
    return apiClient.get(`/ratings/recommendations?limit=${limit}`);
  }
  
  // Update rating notes only
  async updateRatingNotes(paperId: string, notes: string): Promise<Rating> {
    return apiClient.patch<Rating>(`/ratings/paper/${paperId}/notes`, { notes });
  }
  
  // Get rating history for a paper
  async getRatingHistory(paperId: string): Promise<Array<{
    rating: number;
    notes?: string;
    created_at: string;
  }>> {
    return apiClient.get(`/ratings/paper/${paperId}/history`);
  }
}

export const ratingsService = new RatingsService();