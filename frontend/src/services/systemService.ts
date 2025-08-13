import { apiClient } from './api';
import { CacheStats, SystemConfig, RecommendationResult } from '@types';

export class SystemService {
  // Get system configuration
  async getConfig(): Promise<SystemConfig> {
    return apiClient.get<SystemConfig>('/system/config');
  }
  
  // Update system configuration (admin only)
  async updateConfig(config: Partial<SystemConfig>): Promise<SystemConfig> {
    return apiClient.patch<SystemConfig>('/system/config', config);
  }
  
  // Get cache statistics
  async getCacheStats(): Promise<CacheStats> {
    return apiClient.get<CacheStats>('/system/cache/stats');
  }
  
  // Clear cache
  async clearCache(): Promise<{ message: string; cleared_items: number }> {
    return apiClient.post('/system/cache/clear');
  }
  
  // Generate recommendations
  async generateRecommendations(): Promise<RecommendationResult> {
    return apiClient.post<RecommendationResult>('/system/recommendations/generate');
  }
  
  // Get system health
  async getHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    timestamp: string;
    checks: {
      database: boolean;
      embedding_service: boolean;
      api_budget: boolean;
      cache: boolean;
    };
    uptime: number;
  }> {
    return apiClient.get('/system/health');
  }
  
  // Get system metrics
  async getMetrics(): Promise<{
    requests_per_minute: number;
    average_response_time: number;
    error_rate: number;
    cache_hit_rate: number;
    active_embeddings: number;
    memory_usage: number;
    disk_usage: number;
  }> {
    return apiClient.get('/system/metrics');
  }
  
  // Get API usage statistics
  async getApiUsage(): Promise<{
    daily_requests: number;
    monthly_requests: number;
    daily_cost: number;
    monthly_cost: number;
    budget_remaining: number;
    rate_limit_status: {
      requests_per_minute: number;
      limit: number;
      reset_at: string;
    };
    usage_by_endpoint: Record<string, {
      count: number;
      average_time: number;
      error_count: number;
    }>;
  }> {
    return apiClient.get('/system/api-usage');
  }
  
  // Initialize/reset system
  async initializeSystem(): Promise<{ message: string; initialized_components: string[] }> {
    return apiClient.post('/system/initialize');
  }
  
  // Backup system data
  async createBackup(): Promise<{
    backup_id: string;
    size: number;
    created_at: string;
    includes: string[];
  }> {
    return apiClient.post('/system/backup');
  }
  
  // List available backups
  async getBackups(): Promise<Array<{
    id: string;
    size: number;
    created_at: string;
    status: 'complete' | 'partial' | 'failed';
  }>> {
    return apiClient.get('/system/backups');
  }
  
  // Restore from backup
  async restoreBackup(backupId: string): Promise<{ message: string; restored_items: number }> {
    return apiClient.post(`/system/backups/${backupId}/restore`);
  }
  
  // Delete backup
  async deleteBackup(backupId: string): Promise<void> {
    return apiClient.delete(`/system/backups/${backupId}`);
  }
  
  // Get system logs
  async getLogs(params: {
    level?: 'debug' | 'info' | 'warning' | 'error';
    limit?: number;
    since?: string; // ISO date string
  } = {}): Promise<Array<{
    timestamp: string;
    level: string;
    message: string;
    component: string;
    metadata?: Record<string, any>;
  }>> {
    const queryParams = new URLSearchParams();
    
    if (params.level) queryParams.append('level', params.level);
    if (params.limit) queryParams.append('limit', String(params.limit));
    if (params.since) queryParams.append('since', params.since);
    
    return apiClient.get(`/system/logs?${queryParams}`);
  }
  
  // Update user preferences
  async updateUserPreferences(preferences: any): Promise<any> {
    return apiClient.patch('/system/user/preferences', preferences);
  }
  
  // Get user preferences
  async getUserPreferences(): Promise<any> {
    return apiClient.get('/system/user/preferences');
  }
  
  // Test API connection
  async testConnection(): Promise<{ 
    success: boolean; 
    response_time: number; 
    timestamp: string;
  }> {
    const start = Date.now();
    
    try {
      await apiClient.get('/system/ping');
      return {
        success: true,
        response_time: Date.now() - start,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        success: false,
        response_time: Date.now() - start,
        timestamp: new Date().toISOString(),
      };
    }
  }
  
  // Get database stats
  async getDatabaseStats(): Promise<{
    total_papers: number;
    total_ratings: number;
    total_embeddings: number;
    database_size: number;
    last_backup: string;
    index_efficiency: number;
  }> {
    return apiClient.get('/system/database/stats');
  }
  
  // Optimize database
  async optimizeDatabase(): Promise<{
    message: string;
    space_saved: number;
    optimization_time: number;
  }> {
    return apiClient.post('/system/database/optimize');
  }
  
  // Get migration status
  async getMigrationStatus(): Promise<{
    current_version: string;
    latest_version: string;
    pending_migrations: string[];
    last_migration: string;
  }> {
    return apiClient.get('/system/migrations/status');
  }
  
  // Run pending migrations
  async runMigrations(): Promise<{
    completed_migrations: string[];
    failed_migrations: string[];
    total_time: number;
  }> {
    return apiClient.post('/system/migrations/run');
  }
}

export const systemService = new SystemService();