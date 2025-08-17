import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { ApiResponse, ApiError } from '../types/api';

class ApiClient {
  private client: AxiosInstance;
  
  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    this.setupInterceptors();
  }
  
  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add timestamp to prevent caching
        if (config.method === 'get') {
          config.params = {
            ...config.params,
            _t: Date.now(),
          };
        }
        
        // Log outgoing request
        console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, {
          params: config.params,
          data: config.data,
          headers: config.headers,
          timestamp: new Date().toISOString(),
        });
        
        return config;
      },
      (error) => {
        console.error('[API Request Error]', error);
        return Promise.reject(error);
      }
    );
    
    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse<ApiResponse>) => {
        // Log successful response
        console.log(`[API Response] ${response.status} ${response.config.method?.toUpperCase()} ${response.config.url}`, {
          status: response.status,
          correlationId: response.headers['x-correlation-id'],
          responseTime: response.headers['x-response-time'],
          data: response.data,
          timestamp: new Date().toISOString(),
        });
        
        return response;
      },
      (error) => {
        // Enhanced error logging with more context
        const correlationId = error.response?.headers?.['x-correlation-id'];
        const responseTime = error.response?.headers?.['x-response-time'];
        
        console.error(`[API Error] ${error.response?.status || 'Network'} ${error.config?.method?.toUpperCase()} ${error.config?.url}`, {
          status: error.response?.status,
          correlationId,
          responseTime,
          errorData: error.response?.data,
          errorMessage: error.message,
          stack: error.stack,
          timestamp: new Date().toISOString(),
          requestData: error.config?.data,
          requestParams: error.config?.params,
        });
        
        // Create enhanced API error with correlation ID
        const apiError: ApiError = {
          error: error.response?.data?.error || error.name,
          message: this.getErrorMessage(error),
          statusCode: error.response?.status || 500,
          timestamp: new Date().toISOString(),
          correlationId,
          responseTime,
          endpoint: error.config?.url,
          method: error.config?.method?.toUpperCase(),
        };
        
        // Show user-friendly error notification
        this.showErrorNotification(apiError);
        
        return Promise.reject(apiError);
      }
    );
  }
  
  private getErrorMessage(error: any): string {
    // Provide user-friendly error messages
    if (error.code === 'NETWORK_ERROR' || !error.response) {
      return 'Unable to connect to the server. Please check your internet connection and try again.';
    }
    
    const status = error.response?.status;
    const serverMessage = error.response?.data?.message;
    
    switch (status) {
      case 400:
        return serverMessage || 'Invalid request. Please check your input and try again.';
      case 401:
        return 'Authentication required. Please log in and try again.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return 'The requested resource was not found.';
      case 429:
        return 'Too many requests. Please wait a moment and try again.';
      case 500:
        return serverMessage || 'An internal server error occurred. Please try again later.';
      case 502:
      case 503:
      case 504:
        return 'The server is temporarily unavailable. Please try again in a few minutes.';
      default:
        return serverMessage || error.message || 'An unexpected error occurred.';
    }
  }
  
  private showErrorNotification(apiError: ApiError): void {
    // Create user-friendly error notification
    const errorDetails = {
      message: apiError.message,
      correlationId: apiError.correlationId,
      timestamp: apiError.timestamp,
      statusCode: apiError.statusCode,
    };
    
    // Store error for potential debugging
    if (typeof window !== 'undefined') {
      const errors = JSON.parse(localStorage.getItem('api_errors') || '[]');
      errors.push(errorDetails);
      // Keep only last 10 errors
      const recentErrors = errors.slice(-10);
      localStorage.setItem('api_errors', JSON.stringify(recentErrors));
      
      // Dispatch custom event for error handling components
      window.dispatchEvent(new CustomEvent('api-error', {
        detail: errorDetails
      }));
    }
  }
  
  async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<ApiResponse<T>>(url, config);
    return response.data.data;
  }
  
  async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<ApiResponse<T>>(url, data, config);
    return response.data.data;
  }
  
  async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<ApiResponse<T>>(url, data, config);
    return response.data.data;
  }
  
  async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.patch<ApiResponse<T>>(url, data, config);
    return response.data.data;
  }
  
  async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<ApiResponse<T>>(url, config);
    return response.data.data;
  }
  
  // Upload file with progress
  async upload<T = any>(
    url: string, 
    file: File, 
    onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<ApiResponse<T>>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
    
    return response.data.data;
  }
  
  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      console.log('[Health Check] Server is healthy', {
        status: response.status,
        data: response.data,
        timestamp: new Date().toISOString(),
      });
      return true;
    } catch (error) {
      console.error('[Health Check] Server health check failed', {
        error,
        timestamp: new Date().toISOString(),
      });
      return false;
    }
  }
  
  // Get stored API errors for debugging
  getStoredErrors(): any[] {
    if (typeof window === 'undefined') return [];
    return JSON.parse(localStorage.getItem('api_errors') || '[]');
  }
  
  // Clear stored errors
  clearStoredErrors(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('api_errors');
    }
  }
  
  // Get raw axios instance for custom usage
  getClient(): AxiosInstance {
    return this.client;
  }
}

// Create singleton instance
export const apiClient = new ApiClient();

// Export for testing or custom configurations
export { ApiClient };