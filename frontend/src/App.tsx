import { Routes, Route } from 'react-router-dom';
import { Layout } from '@components/common/Layout';
import { DashboardPage } from '@pages/DashboardPage';
import { PapersPage } from '@pages/PapersPage';
import { CollectPapersPage } from '@pages/CollectPapersPage';
import { RatingPage } from '@pages/RatingPage';
import { SearchPage } from '@pages/SearchPage';
import { AnalyticsPage } from '@pages/AnalyticsPage';
import { SettingsPage } from '@pages/SettingsPage';
import { ErrorBoundary } from '@components/common/ErrorBoundary';

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="papers" element={<PapersPage />} />
          <Route path="collect" element={<CollectPapersPage />} />
          <Route path="rate" element={<RatingPage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </ErrorBoundary>
  );
}
