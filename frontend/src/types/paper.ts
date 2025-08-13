export interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  category: string;
  published_date: string;
  arxiv_url: string;
  pdf_url: string;
  current_score?: number;
  rating?: number;
  notes?: string;
  created_at?: string;
  updated_at?: string;
}

export interface PaperFilters {
  category?: string;
  rating?: number | 'unrated';
  dateRange?: {
    start: Date;
    end: Date;
  };
  searchQuery?: string;
}

export interface PaperSearchParams {
  query: string;
  filters?: PaperFilters;
  limit?: number;
  offset?: number;
  sortBy?: 'relevance' | 'date' | 'rating' | 'score';
  sortOrder?: 'asc' | 'desc';
}

export interface PaperSearchResult {
  papers: Paper[];
  total: number;
  hasMore: boolean;
  facets?: {
    categories: Record<string, number>;
    ratings: Record<number, number>;
  };
}