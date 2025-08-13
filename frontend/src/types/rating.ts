export interface Rating {
  id: string;
  paper_id: string;
  rating: number;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface RatingUpdate {
  paper_id: string;
  rating: number;
  notes?: string;
}

export interface RatingStats {
  total_ratings: number;
  average_rating: number;
  distribution: Record<number, number>;
  completion_rate: number;
}

export interface CategoryRating {
  category: string;
  average_rating: number;
  count: number;
  percentage: number;
}

export interface RatingTimeline {
  date: string;
  daily_ratings: number;
  cumulative_ratings: number;
  average_rating: number;
}