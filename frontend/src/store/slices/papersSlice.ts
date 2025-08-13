import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Paper, PaperFilters, PaperSearchParams, PaperSearchResult } from '@types';
import { papersService } from '@services/papersService';

interface PapersState {
  items: Paper[];
  searchResults: Paper[];
  recommendations: Paper[];
  filters: PaperFilters;
  searchParams: PaperSearchParams | null;
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
        state.recommendations = action.payload.papers;
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
        state.recommendations = action.payload.papers;
      })
      .addCase(generateRecommendations.rejected, (state, action) => {
        state.loading.recommendations = false;
        state.error = action.error.message || 'Failed to generate recommendations';
      });
  },
});

export const {
  setFilters,
  clearFilters,
  clearSearchResults,
  updatePaper,
  clearError,
} = papersSlice.actions;

export default papersSlice.reducer;