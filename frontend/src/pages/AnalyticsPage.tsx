import React, { useState, useEffect } from 'react';
import { analyticsService, type AnalyticsOverview } from '../services/analyticsService';

export const AnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await analyticsService.getOverview();
      setAnalytics(data);
    } catch (err) {
      setError('Failed to load analytics data');
      console.error('Analytics error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRatingStars = (rating: number) => {
    return '⭐'.repeat(rating);
  };

  const getProgressBarColor = (percentage: number) => {
    if (percentage >= 80) return 'bg-green-500';
    if (percentage >= 60) return 'bg-yellow-500';
    if (percentage >= 40) return 'bg-orange-500';
    return 'bg-red-500';
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">📊 Analytics</h1>
          <p className="mt-2 text-secondary-600">
            View your reading patterns and recommendation analytics.
          </p>
        </div>
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-secondary-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-secondary-900">📊 Analytics</h1>
          <p className="mt-2 text-secondary-600">
            View your reading patterns and recommendation analytics.
          </p>
        </div>
        <div className="text-center py-12">
          <div className="text-6xl mb-4">⚠️</div>
          <h3 className="text-lg font-medium text-red-900 mb-2">Error Loading Analytics</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={loadAnalytics}
            className="bg-primary-600 text-white px-4 py-2 rounded hover:bg-primary-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!analytics) {
    return null;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">📊 Analytics</h1>
        <p className="mt-2 text-secondary-600">
          View your reading patterns and recommendation analytics.
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="text-2xl mr-3">📚</div>
            <div>
              <p className="text-sm font-medium text-secondary-600">Total Papers</p>
              <p className="text-3xl font-bold text-secondary-900">{analytics.overview.total_papers}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="text-2xl mr-3">⭐</div>
            <div>
              <p className="text-sm font-medium text-secondary-600">Rated Papers</p>
              <p className="text-3xl font-bold text-secondary-900">{analytics.overview.rated_papers}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="text-2xl mr-3">📊</div>
            <div>
              <p className="text-sm font-medium text-secondary-600">Completion Rate</p>
              <p className="text-3xl font-bold text-secondary-900">{analytics.overview.completion_rate}%</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <div className="text-2xl mr-3">📈</div>
            <div>
              <p className="text-sm font-medium text-secondary-600">Average Rating</p>
              <p className="text-3xl font-bold text-secondary-900">{analytics.overview.average_rating}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Rating Distribution */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-secondary-900 mb-4">Rating Distribution</h2>
        <div className="space-y-4">
          {Object.entries(analytics.rating_distribution)
            .sort(([a], [b]) => parseInt(b) - parseInt(a))
            .map(([rating, count]) => {
              const percentage = analytics.overview.rated_papers > 0 
                ? (count / analytics.overview.rated_papers * 100) 
                : 0;
              
              return (
                <div key={rating} className="flex items-center">
                  <div className="w-20 text-sm font-medium text-secondary-700">
                    {getRatingStars(parseInt(rating))} ({rating})
                  </div>
                  <div className="flex-1 mx-4">
                    <div className="bg-secondary-200 rounded-full h-3">
                      <div
                        className={`${getProgressBarColor(percentage)} h-3 rounded-full transition-all duration-300`}
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="w-20 text-sm text-secondary-600 text-right">
                    {count} ({percentage.toFixed(1)}%)
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-secondary-900 mb-4">Papers by Category</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-secondary-200">
            <thead className="bg-secondary-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                  Total Papers
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                  Rated Papers
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                  Average Rating
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-secondary-500 uppercase tracking-wider">
                  Percentage
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-secondary-200">
              {Object.entries(analytics.category_breakdown)
                .sort(([, a], [, b]) => b.total_papers - a.total_papers)
                .map(([category, stats]) => (
                  <tr key={category} className="hover:bg-secondary-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-secondary-900">
                      {category}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600">
                      {stats.total_papers}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600">
                      {stats.rated_papers}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600">
                      {stats.average_rating > 0 ? `${stats.average_rating} ⭐` : 'No ratings'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-secondary-600">
                      {stats.percentage}%
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary Insights */}
      {analytics.summary.most_common_rating && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-secondary-900 mb-4">Insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="text-blue-800 font-medium">Most Common Rating</div>
              <div className="text-2xl font-bold text-blue-900 mt-1">
                {getRatingStars(analytics.summary.most_common_rating)} ({analytics.summary.most_common_rating})
              </div>
            </div>
            
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="text-green-800 font-medium">Total Categories</div>
              <div className="text-2xl font-bold text-green-900 mt-1">
                {analytics.summary.total_categories}
              </div>
            </div>
            
            {analytics.summary.best_rated_category && (
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="text-purple-800 font-medium">Best Rated Category</div>
                <div className="text-lg font-bold text-purple-900 mt-1">
                  {analytics.summary.best_rated_category}
                </div>
                <div className="text-sm text-purple-700">
                  {analytics.category_breakdown[analytics.summary.best_rated_category]?.average_rating.toFixed(2)} ⭐
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Refresh Button */}
      <div className="text-center">
        <button
          onClick={loadAnalytics}
          className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
        >
          Refresh Analytics
        </button>
      </div>
    </div>
  );
};