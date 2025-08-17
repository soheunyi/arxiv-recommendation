import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout } from '@components/common/Layout';
import { DashboardPage } from '@pages/DashboardPage';
import { PapersPage } from '@pages/PapersPage';
import { PaperDetailPage } from '@pages/PaperDetailPage';
import { CollectPapersPage } from '@pages/CollectPapersPage';
import { RatingPage } from '@pages/RatingPage';
import { SearchPage } from '@pages/SearchPage';
import { AnalyticsPage } from '@pages/AnalyticsPage';
import { ScoredPapersPage } from '@pages/ScoredPapersPage';
import { SettingsPage } from '@pages/SettingsPage';
import { ErrorBoundary } from '@components/common/ErrorBoundary';
import { ErrorDisplay } from '@components/common/ErrorDisplay';
import { DebugPanel } from '@components/common/DebugPanel';
import { useErrorHandler } from '@hooks/useErrorHandler';

export default function App() {
  const { currentError, dismissError } = useErrorHandler();
  const [showDebugPanel, setShowDebugPanel] = useState(false);

  // Enable debug panel with Ctrl+Shift+D
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey && event.shiftKey && event.key === 'D') {
        event.preventDefault();
        setShowDebugPanel(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="papers" element={<PapersPage />} />
          <Route path="papers/:paperId" element={<PaperDetailPage />} />
          <Route path="collect" element={<CollectPapersPage />} />
          <Route path="rate" element={<RatingPage />} />
          <Route path="search" element={<SearchPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="scores" element={<ScoredPapersPage />} />
          <Route path="settings" element={<SettingsPage />} />
        </Route>
      </Routes>

      {/* Global Error Display */}
      <ErrorDisplay 
        error={currentError} 
        onDismiss={dismissError}
        showDetails={process.env.NODE_ENV === 'development'}
      />

      {/* Debug Panel */}
      <DebugPanel 
        isOpen={showDebugPanel}
        onClose={() => setShowDebugPanel(false)}
      />

      {/* Debug Panel Trigger (Development Only) */}
      {process.env.NODE_ENV === 'development' && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: 999,
        }}>
          <button
            onClick={() => setShowDebugPanel(true)}
            style={{
              background: '#3182ce',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '48px',
              height: '48px',
              fontSize: '20px',
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
            }}
            title="Open Debug Panel (Ctrl+Shift+D)"
          >
            🐛
          </button>
        </div>
      )}
    </ErrorBoundary>
  );
}
