import { apiClient } from './api';
import { 
  Paper, 
  PaperSearchParams, 
  PaperSearchResult, 
  PaperFilters,
  PaginatedResponse,
  RecommendationResult 
} from '@types';

export class PapersService {
  // Get paginated papers with filters
  async getPapers(params: {
    page?: number;
    limit?: number;
    filters?: PaperFilters;
  }): Promise<PaginatedResponse<Paper>> {
    const queryParams = new URLSearchParams({
      page: String(params.page || 1),
      limit: String(params.limit || 20),
    });
    
    // Add filters to query params
    if (params.filters) {
      if (params.filters.category) {
        queryParams.append('category', params.filters.category);
      }
      if (params.filters.rating !== undefined) {
        if (params.filters.rating === 'unrated') {
          queryParams.append('rating', 'unrated');
        } else {
          queryParams.append('rating', String(params.filters.rating));
        }
      }
      if (params.filters.searchQuery) {
        queryParams.append('search', params.filters.searchQuery);
      }
      if (params.filters.dateRange) {
        queryParams.append('start_date', params.filters.dateRange.start.toISOString());
        queryParams.append('end_date', params.filters.dateRange.end.toISOString());
      }
    }
    
    return apiClient.get<PaginatedResponse<Paper>>(`/papers?${queryParams}`);
  }
  
  // Get all papers (for admin/bulk operations)
  async getAllPapers(): Promise<Paper[]> {
    return apiClient.get<Paper[]>('/papers/all');
  }
  
  // Get papers for rating interface with filters
  async getPapersForRating(filters: {
    category?: string;
    rating?: string;
    limit?: number;
  }): Promise<Paper[]> {
    const queryParams = new URLSearchParams();
    
    if (filters.category && filters.category !== 'All') {
      queryParams.append('category', filters.category);
    }
    if (filters.rating && filters.rating !== 'All') {
      queryParams.append('rating_filter', filters.rating);
    }
    if (filters.limit) {
      queryParams.append('limit', String(filters.limit));
    }
    
    return apiClient.get<Paper[]>(`/papers/for-rating?${queryParams}`);
  }
  
  // Search papers
  async searchPapers(searchParams: PaperSearchParams): Promise<PaperSearchResult> {
    return apiClient.post<PaperSearchResult>('/papers/search', searchParams);
  }
  
  // Get recent papers
  async getRecentPapers(limit: number = 20): Promise<Paper[]> {
    return apiClient.get<Paper[]>(`/papers/recent?limit=${limit}`);
  }
  
  // Get recommendations
  async getRecommendations(limit: number = 10): Promise<RecommendationResult> {
    return apiClient.get<RecommendationResult>(`/recommendations/recent?limit=${limit}`);
  }
  
  // Generate new recommendations
  async generateRecommendations(): Promise<RecommendationResult> {
    return apiClient.post<RecommendationResult>('/recommendations/generate');
  }
  
  // Get single paper by ID
  async getPaper(id: string): Promise<Paper> {
    return apiClient.get<Paper>(`/papers/${id}`);
  }
  
  // Update paper (admin only)
  async updatePaper(id: string, updates: Partial<Paper>): Promise<Paper> {
    return apiClient.patch<Paper>(`/papers/${id}`, updates);
  }
  
  // Delete paper (admin only)
  async deletePaper(id: string): Promise<void> {
    return apiClient.delete(`/papers/${id}`);
  }
  
  // Bulk operations
  async bulkUpdatePapers(updates: Array<{ id: string; updates: Partial<Paper> }>): Promise<Paper[]> {
    return apiClient.post<Paper[]>('/papers/bulk-update', { updates });
  }
  
  // Get paper categories
  async getCategories(): Promise<string[]> {
    return apiClient.get<string[]>('/papers/categories');
  }
  
  // Get paper stats
  async getPaperStats(): Promise<{
    total: number;
    by_category: Record<string, number>;
    recent_additions: number;
    avg_rating: number;
  }> {
    return apiClient.get('/papers/stats');
  }
  
  // Export papers (for backup/analysis)
  async exportPapers(format: 'json' | 'csv' = 'json'): Promise<Blob> {
    const response = await apiClient.getClient().get(`/papers/export?format=${format}`, {
      responseType: 'blob',
    });
    
    return response.data;
  }
  
  // Import papers (for bulk upload)
  async importPapers(file: File, onProgress?: (progress: number) => void): Promise<{
    imported: number;
    skipped: number;
    errors: string[];
  }> {
    return apiClient.upload('/papers/import', file, onProgress);
  }
}

export const papersService = new PapersService();