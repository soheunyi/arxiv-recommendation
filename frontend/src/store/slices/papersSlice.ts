import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Paper, PaperFilters, PaperSearchParams, PaperSearchResult } from '../../types';
import { papersService } from '@services/papersService';

interface CollectionStatus {
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
}

interface PapersState {
  items: Paper[];
  searchResults: Paper[];
  recommendations: Paper[];
  filters: PaperFilters;
  searchParams: PaperSearchParams | null;
  collection: {
    current: CollectionStatus | null;
    isModalOpen: boolean;
  };
  llmProviders: {
    current: string;
    available: Record<string, any>;
    recommendation: string;
  } | null;
  pagination: {
    currentPage: number;
    totalPages: number;
    hasMore: boolean;
    total: number;
  };
  loading: {
    fetch: boolean;
    search: boolean;
    recommendations: boolean;
    collection: boolean;
    providers: boolean;
  };
  error: string | null;
  lastUpdated: string | null;
}

const initialState: PapersState = {
  items: [],
  searchResults: [],
  recommendations: [],
  filters: {},
  searchParams: null,
  collection: {
    current: null,
    isModalOpen: false,
  },
  llmProviders: null,
  pagination: {
    currentPage: 1,
    totalPages: 1,
    hasMore: false,
    total: 0,
  },
  loading: {
    fetch: false,
    search: false,
    recommendations: false,
    collection: false,
    providers: false,
  },
  error: null,
  lastUpdated: null,
};

// Async thunks
export const fetchPapers = createAsyncThunk(
  'papers/fetchPapers',
  async (params: { page?: number; limit?: number; filters?: PaperFilters }) => {
    const response = await papersService.getPapers(params);
    return response;
  }
);

export const searchPapers = createAsyncThunk(
  'papers/searchPapers',
  async (searchParams: PaperSearchParams) => {
    const response = await papersService.searchPapers(searchParams);
    return { searchParams, result: response };
  }
);

export const fetchRecommendations = createAsyncThunk(
  'papers/fetchRecommendations',
  async () => {
    const response = await papersService.getRecommendations();
    return response;
  }
);

export const generateRecommendations = createAsyncThunk(
  'papers/generateRecommendations',
  async () => {
    const response = await papersService.generateRecommendations();
    return response;
  }
);

export const startCollection = createAsyncThunk(
  'papers/startCollection',
  async (params: { 
    keyword: string; 
    max_papers?: number; 
    clean_db?: boolean; 
    llm_provider?: string;
    date_from?: string;
    date_to?: string;
    use_collaboration?: boolean;
    collaboration_strategy?: string;
  }) => {
    const response = await papersService.startCollection(params);
    return response;
  }
);

export const fetchCollectionStatus = createAsyncThunk(
  'papers/fetchCollectionStatus',
  async (collectionId: string) => {
    const response = await papersService.getCollectionStatus(collectionId);
    return response;
  }
);

export const fetchLLMProviders = createAsyncThunk(
  'papers/fetchLLMProviders',
  async () => {
    const response = await papersService.getLLMProviders();
    return response;
  }
);

const papersSlice = createSlice({
  name: 'papers',
  initialState,
  reducers: {
    setFilters: (state, action: PayloadAction<PaperFilters>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    clearFilters: (state) => {
      state.filters = {};
    },
    clearSearchResults: (state) => {
      state.searchResults = [];
      state.searchParams = null;
    },
    updatePaper: (state, action: PayloadAction<Partial<Paper> & { id: string }>) => {
      const { id, ...updates } = action.payload;
      
      // Update in items
      const itemIndex = state.items.findIndex(paper => paper.id === id);
      if (itemIndex !== -1) {
        state.items[itemIndex] = { ...state.items[itemIndex], ...updates };
      }
      
      // Update in search results
      const searchIndex = state.searchResults.findIndex(paper => paper.id === id);
      if (searchIndex !== -1) {
        state.searchResults[searchIndex] = { ...state.searchResults[searchIndex], ...updates };
      }
      
      // Update in recommendations
      const recIndex = state.recommendations.findIndex(paper => paper.id === id);
      if (recIndex !== -1) {
        state.recommendations[recIndex] = { ...state.recommendations[recIndex], ...updates };
      }
    },
    clearError: (state) => {
      state.error = null;
    },
    openCollectionModal: (state) => {
      state.collection.isModalOpen = true;
    },
    closeCollectionModal: (state) => {
      state.collection.isModalOpen = false;
    },
    clearCollection: (state) => {
      state.collection.current = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch papers
    builder
      .addCase(fetchPapers.pending, (state) => {
        state.loading.fetch = true;
        state.error = null;
      })
      .addCase(fetchPapers.fulfilled, (state, action) => {
        state.loading.fetch = false;
        state.items = action.payload.items;
        state.pagination = {
          currentPage: action.payload.page,
          totalPages: action.payload.totalPages,
          hasMore: action.payload.hasNext,
          total: action.payload.total,
        };
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchPapers.rejected, (state, action) => {
        state.loading.fetch = false;
        state.error = action.error.message || 'Failed to fetch papers';
      });

    // Search papers
    builder
      .addCase(searchPapers.pending, (state) => {
        state.loading.search = true;
        state.error = null;
      })
      .addCase(searchPapers.fulfilled, (state, action) => {
        state.loading.search = false;
        state.searchResults = action.payload.result.papers;
        state.searchParams = action.payload.searchParams;
      })
      .addCase(searchPapers.rejected, (state, action) => {
        state.loading.search = false;
        state.error = action.error.message || 'Search failed';
      });

    // Fetch recommendations
    builder
      .addCase(fetchRecommendations.pending, (state) => {
        state.loading.recommendations = true;
        state.error = null;
      })
      .addCase(fetchRecommendations.fulfilled, (state, action) => {
        state.loading.recommendations = false;
        state.recommendations = action.payload.recommendations || action.payload.papers || [];
      })
      .addCase(fetchRecommendations.rejected, (state, action) => {
        state.loading.recommendations = false;
        state.error = action.error.message || 'Failed to fetch recommendations';
      });

    // Generate recommendations
    builder
      .addCase(generateRecommendations.pending, (state) => {
        state.loading.recommendations = true;
        state.error = null;
      })
      .addCase(generateRecommendations.fulfilled, (state, action) => {
        state.loading.recommendations = false;
        state.recommendations = action.payload.recommendations || action.payload.papers || [];
      })
      .addCase(generateRecommendations.rejected, (state, action) => {
        state.loading.recommendations = false;
        state.error = action.error.message || 'Failed to generate recommendations';
      });

    // Start collection
    builder
      .addCase(startCollection.pending, (state) => {
        state.loading.collection = true;
        state.error = null;
      })
      .addCase(startCollection.fulfilled, (state, action) => {
        state.loading.collection = false;
        state.collection.current = {
          id: action.payload.collection_id,
          keyword: action.payload.keyword,
          status: 'running',
          progress: 0,
          papers_found: 0,
          total_queries: 0,
          current_query: 'Starting collection...',
          created_at: new Date().toISOString(),
        };
        state.collection.isModalOpen = false;
      })
      .addCase(startCollection.rejected, (state, action) => {
        state.loading.collection = false;
        state.error = action.error.message || 'Failed to start collection';
      });

    // Fetch collection status
    builder
      .addCase(fetchCollectionStatus.pending, (state) => {
        // Don't set loading for status polling
      })
      .addCase(fetchCollectionStatus.fulfilled, (state, action) => {
        state.collection.current = action.payload;
      })
      .addCase(fetchCollectionStatus.rejected, (state, action) => {
        // Silent fail for status polling
        console.error('Failed to fetch collection status:', action.error.message);
      });

    // Fetch LLM providers
    builder
      .addCase(fetchLLMProviders.pending, (state) => {
        state.loading.providers = true;
        state.error = null;
      })
      .addCase(fetchLLMProviders.fulfilled, (state, action) => {
        state.loading.providers = false;
        state.llmProviders = {
          current: action.payload.current_provider,
          available: action.payload.providers,
          recommendation: action.payload.recommendation,
        };
      })
      .addCase(fetchLLMProviders.rejected, (state, action) => {
        state.loading.providers = false;
        state.error = action.error.message || 'Failed to fetch LLM providers';
      });
  },
});

export const {
  setFilters,
  clearFilters,
  clearSearchResults,
  updatePaper,
  clearError,
  openCollectionModal,
  closeCollectionModal,
  clearCollection,
} = papersSlice.actions;

export default papersSlice.reducer;