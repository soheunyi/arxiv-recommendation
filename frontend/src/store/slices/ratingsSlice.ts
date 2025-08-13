import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Rating, RatingUpdate, RatingStats } from '@types';
import { ratingsService } from '@services/ratingsService';

interface RatingsState {
  userRatings: Record<string, Rating>;
  stats: RatingStats | null;
  pendingUpdates: string[];
  loading: {
    update: boolean;
    fetch: boolean;
    stats: boolean;
  };
  error: string | null;
  lastSynced: string | null;
}

const initialState: RatingsState = {
  userRatings: {},
  stats: null,
  pendingUpdates: [],
  loading: {
    update: false,
    fetch: false,
    stats: false,
  },
  error: null,
  lastSynced: null,
};

// Async thunks
export const fetchUserRatings = createAsyncThunk(
  'ratings/fetchUserRatings',
  async () => {
    const response = await ratingsService.getUserRatings();
    return response;
  }
);

export const updateRating = createAsyncThunk(
  'ratings/updateRating',
  async (ratingUpdate: RatingUpdate, { dispatch }) => {
    // Optimistic update
    dispatch(setOptimisticRating(ratingUpdate));
    
    try {
      const response = await ratingsService.updateRating(ratingUpdate);
      return response;
    } catch (error) {
      // Revert optimistic update on error
      dispatch(revertOptimisticRating(ratingUpdate.paper_id));
      throw error;
    }
  }
);

export const fetchRatingStats = createAsyncThunk(
  'ratings/fetchStats',
  async () => {
    const response = await ratingsService.getRatingStats();
    return response;
  }
);

export const batchUpdateRatings = createAsyncThunk(
  'ratings/batchUpdate',
  async (updates: RatingUpdate[]) => {
    const response = await ratingsService.batchUpdateRatings(updates);
    return response;
  }
);

const ratingsSlice = createSlice({
  name: 'ratings',
  initialState,
  reducers: {
    setOptimisticRating: (state, action: PayloadAction<RatingUpdate>) => {
      const { paper_id, rating, notes } = action.payload;
      
      // Create optimistic rating
      const optimisticRating: Rating = {
        id: `temp-${paper_id}`,
        paper_id,
        rating,
        notes,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      
      state.userRatings[paper_id] = optimisticRating;
      
      // Add to pending updates
      if (!state.pendingUpdates.includes(paper_id)) {
        state.pendingUpdates.push(paper_id);
      }
    },
    
    revertOptimisticRating: (state, action: PayloadAction<string>) => {
      const paperId = action.payload;
      
      // Remove optimistic rating
      delete state.userRatings[paperId];
      
      // Remove from pending updates
      state.pendingUpdates = state.pendingUpdates.filter(id => id !== paperId);
    },
    
    clearPendingUpdate: (state, action: PayloadAction<string>) => {
      const paperId = action.payload;
      state.pendingUpdates = state.pendingUpdates.filter(id => id !== paperId);
    },
    
    setRating: (state, action: PayloadAction<Rating>) => {
      const rating = action.payload;
      state.userRatings[rating.paper_id] = rating;
    },
    
    removeRating: (state, action: PayloadAction<string>) => {
      const paperId = action.payload;
      delete state.userRatings[paperId];
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    // Offline support
    queueRatingUpdate: (state, action: PayloadAction<RatingUpdate>) => {
      const { paper_id } = action.payload;
      if (!state.pendingUpdates.includes(paper_id)) {
        state.pendingUpdates.push(paper_id);
      }
    },
  },
  extraReducers: (builder) => {
    // Fetch user ratings
    builder
      .addCase(fetchUserRatings.pending, (state) => {
        state.loading.fetch = true;
        state.error = null;
      })
      .addCase(fetchUserRatings.fulfilled, (state, action) => {
        state.loading.fetch = false;
        state.userRatings = action.payload.reduce((acc, rating) => {
          acc[rating.paper_id] = rating;
          return acc;
        }, {} as Record<string, Rating>);
        state.lastSynced = new Date().toISOString();
      })
      .addCase(fetchUserRatings.rejected, (state, action) => {
        state.loading.fetch = false;
        state.error = action.error.message || 'Failed to fetch ratings';
      });

    // Update rating
    builder
      .addCase(updateRating.pending, (state, action) => {
        state.loading.update = true;
        const paperId = action.meta.arg.paper_id;
        if (!state.pendingUpdates.includes(paperId)) {
          state.pendingUpdates.push(paperId);
        }
      })
      .addCase(updateRating.fulfilled, (state, action) => {
        state.loading.update = false;
        const rating = action.payload;
        state.userRatings[rating.paper_id] = rating;
        state.pendingUpdates = state.pendingUpdates.filter(id => id !== rating.paper_id);
      })
      .addCase(updateRating.rejected, (state, action) => {
        state.loading.update = false;
        state.error = action.error.message || 'Failed to update rating';
      });

    // Fetch rating stats
    builder
      .addCase(fetchRatingStats.pending, (state) => {
        state.loading.stats = true;
      })
      .addCase(fetchRatingStats.fulfilled, (state, action) => {
        state.loading.stats = false;
        state.stats = action.payload;
      })
      .addCase(fetchRatingStats.rejected, (state, action) => {
        state.loading.stats = false;
        state.error = action.error.message || 'Failed to fetch rating stats';
      });

    // Batch update ratings
    builder
      .addCase(batchUpdateRatings.pending, (state) => {
        state.loading.update = true;
      })
      .addCase(batchUpdateRatings.fulfilled, (state, action) => {
        state.loading.update = false;
        action.payload.forEach(rating => {
          state.userRatings[rating.paper_id] = rating;
        });
        // Clear all pending updates on successful batch
        state.pendingUpdates = [];
      })
      .addCase(batchUpdateRatings.rejected, (state, action) => {
        state.loading.update = false;
        state.error = action.error.message || 'Failed to batch update ratings';
      });
  },
});

export const {
  setOptimisticRating,
  revertOptimisticRating,
  clearPendingUpdate,
  setRating,
  removeRating,
  clearError,
  queueRatingUpdate,
} = ratingsSlice.actions;

export default ratingsSlice.reducer;