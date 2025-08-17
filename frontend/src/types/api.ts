export interface ApiResponse<T = any> {
  data: T;
  success: boolean;
  message?: string;
  timestamp: string;
}

export interface ApiError {
  error: string;
  message: string;
  statusCode: number;
  timestamp: string;
  correlationId?: string;
  responseTime?: string;
  endpoint?: string;
  method?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

export interface CacheStats {
  hit_rate: number;
  total_embeddings: number;
  cache_size_mb: number;
  daily_cost: number;
  last_updated: string;
}

export interface SystemConfig {
  openai_api_key_set: boolean;
  embedding_model: string;
  arxiv_categories: string[];
  max_daily_papers: number;
  budget_limit: number;
  database_path: string;
}

export interface RecommendationResult {
  papers: any[];
  generated_at: string;
  model_version: string;
  parameters: Record<string, any>;
}