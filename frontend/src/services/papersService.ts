import { apiClient } from './api';
import { 
  Paper, 
  PaperSearchParams, 
  PaperSearchResult, 
  PaperFilters,
  PaginatedResponse,
  RecommendationResult 
} from '../../types';

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
  async getRecentPapers(limit: number = 20, sortBy: string = "created_at"): Promise<Paper[]> {
    return apiClient.get<Paper[]>(`/papers/recent?limit=${limit}&sort_by=${sortBy}`);
  }
  
  // Get papers sorted by recommendation score
  async getPapersByScore(limit: number = 20): Promise<Paper[]> {
    return this.getRecentPapers(limit, "score");
  }
  
  // Get recommendations
  async getRecommendations(limit: number = 10): Promise<RecommendationResult> {
    const response = await apiClient.get<any>(`/recommendations/recent?limit=${limit}`);
    // Handle the wrapped response format from backend
    return response.data || response;
  }
  
  // Generate new recommendations
  async generateRecommendations(): Promise<RecommendationResult> {
    const response = await apiClient.post<any>('/recommendations/generate');
    // Handle the wrapped response format from backend
    return response.data || response;
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

  // Collection management
  async startCollection(params: {
    keyword: string;
    max_papers?: number;
    clean_db?: boolean;
    llm_provider?: string;
    date_from?: string;
    date_to?: string;
    use_collaboration?: boolean;
    collaboration_strategy?: string;
  }): Promise<{
    collection_id: string;
    status: string;
    keyword: string;
    estimated_time: number;
  }> {
    return apiClient.post('/collection/start', params);
  }

  // Get available LLM providers
  async getLLMProviders(): Promise<{
    current_provider: string;
    providers: Record<string, any>;
    recommendation: string;
  }> {
    return apiClient.get('/collection/providers');
  }

  async getCollectionStatus(collectionId: string): Promise<{
    id: string;
    keyword: string;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    papers_found: number;
    total_queries: number;
    current_query: string;
    created_at: string;
    completed_at?: string;
    error_message?: string;
  }> {
    return apiClient.get(`/collection/status/${collectionId}`);
  }

  // Collaboration services
  async generateQueriesCollaborative(params: {
    topic: string;
    max_queries?: number;
    date_from?: string;
    date_to?: string;
    strategy?: string;
    quality_threshold?: number;
  }): Promise<{
    queries: any;
    collaboration_info: {
      primary_provider: string;
      secondary_provider?: string;
      strategy_used: string;
      quality_score: number;
      cost_estimate: number;
      execution_time: number;
      has_secondary_result: boolean;
    };
    alternative_queries?: any;
  }> {
    return apiClient.post('/collaboration/generate-queries', params);
  }

  async getCollaborationStrategies(): Promise<{
    strategies: Record<string, {
      name: string;
      description: string;
      use_case: string;
      cost_impact: string;
      quality_impact: string;
    }>;
    default: string;
    recommendation: string;
  }> {
    return apiClient.get('/collaboration/strategies');
  }

  async getCollaborationUsageStats(): Promise<{
    providers: Record<string, { requests: number; total_cost: number }>;
    total_requests: number;
    total_cost: number;
    budget_used_percentage: number;
    cost_savings: {
      absolute_savings: number;
      percentage_savings: number;
    };
  }> {
    return apiClient.get('/collaboration/usage-stats');
  }

  async switchProvider(provider: string): Promise<{
    switched: boolean;
    new_provider: string;
    note: string;
  }> {
    return apiClient.post('/collaboration/switch-provider', { provider });
  }

  // Manually add a paper by ArXiv ID
  async addManualPaper(request: {
    arxiv_id: string;
    category?: string;
  }): Promise<{
    success: boolean;
    data: {
      paper: Paper;
      manually_added: boolean;
      already_exists?: boolean;
      arxiv_id: string;
    };
    message: string;
  }> {
    return apiClient.post('/papers/manual-add', request);
  }
}

export const papersService = new PapersService();