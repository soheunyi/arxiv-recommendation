import React, { useState, useEffect } from 'react';
import { apiClient } from '../../services/api';
import { ApiError } from '../../types/api';

interface DebugPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ isOpen, onClose }) => {
  const [apiErrors, setApiErrors] = useState<ApiError[]>([]);
  const [serverHealth, setServerHealth] = useState<{
    healthy: boolean;
    lastChecked: string;
    checking: boolean;
  }>({
    healthy: false,
    lastChecked: '',
    checking: false,
  });

  useEffect(() => {
    if (isOpen) {
      loadApiErrors();
      checkServerHealth();
    }
  }, [isOpen]);

  const loadApiErrors = () => {
    const errors = apiClient.getStoredErrors();
    setApiErrors(errors);
  };

  const checkServerHealth = async () => {
    setServerHealth(prev => ({ ...prev, checking: true }));
    
    try {
      const healthy = await apiClient.healthCheck();
      setServerHealth({
        healthy,
        lastChecked: new Date().toLocaleTimeString(),
        checking: false,
      });
    } catch {
      setServerHealth({
        healthy: false,
        lastChecked: new Date().toLocaleTimeString(),
        checking: false,
      });
    }
  };

  const clearErrors = () => {
    apiClient.clearStoredErrors();
    setApiErrors([]);
  };

  const copyErrorToClipboard = async (error: ApiError) => {
    const errorText = JSON.stringify(error, null, 2);
    try {
      await navigator.clipboard.writeText(errorText);
      alert('Error details copied to clipboard');
    } catch {
      console.log('Copy failed', errorText);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="debug-panel">
      <div className="debug-panel__overlay" onClick={onClose} />
      <div className="debug-panel__content">
        <div className="debug-panel__header">
          <h3>Debug Panel</h3>
          <button onClick={onClose} className="debug-panel__close">✕</button>
        </div>

        <div className="debug-panel__section">
          <h4>Server Health</h4>
          <div className="debug-panel__health">
            <div className={`debug-panel__status debug-panel__status--${serverHealth.healthy ? 'healthy' : 'unhealthy'}`}>
              {serverHealth.checking ? '⏳ Checking...' : serverHealth.healthy ? '✅ Healthy' : '❌ Unhealthy'}
            </div>
            <div className="debug-panel__health-info">
              Last checked: {serverHealth.lastChecked}
            </div>
            <button 
              onClick={checkServerHealth} 
              disabled={serverHealth.checking}
              className="debug-panel__button"
            >
              Refresh Health Check
            </button>
          </div>
        </div>

        <div className="debug-panel__section">
          <div className="debug-panel__section-header">
            <h4>Recent API Errors ({apiErrors.length})</h4>
            <div className="debug-panel__actions">
              <button onClick={loadApiErrors} className="debug-panel__button debug-panel__button--small">
                Refresh
              </button>
              <button onClick={clearErrors} className="debug-panel__button debug-panel__button--small">
                Clear All
              </button>
            </div>
          </div>

          {apiErrors.length === 0 ? (
            <div className="debug-panel__empty">No recent errors</div>
          ) : (
            <div className="debug-panel__errors">
              {apiErrors.map((error, index) => (
                <div key={index} className="debug-panel__error">
                  <div className="debug-panel__error-header">
                    <span className="debug-panel__error-status">{error.statusCode}</span>
                    <span className="debug-panel__error-method">{error.method}</span>
                    <span className="debug-panel__error-endpoint">{error.endpoint}</span>
                    <button 
                      onClick={() => copyErrorToClipboard(error)}
                      className="debug-panel__copy"
                      title="Copy error details"
                    >
                      📋
                    </button>
                  </div>
                  <div className="debug-panel__error-message">{error.message}</div>
                  <div className="debug-panel__error-meta">
                    <div>ID: <code>{error.correlationId || 'N/A'}</code></div>
                    <div>Time: {error.responseTime || 'N/A'}</div>
                    <div>Timestamp: {new Date(error.timestamp).toLocaleString()}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="debug-panel__section">
          <h4>Environment Info</h4>
          <div className="debug-panel__env">
            <div>User Agent: {navigator.userAgent}</div>
            <div>Viewport: {window.innerWidth}x{window.innerHeight}</div>
            <div>Connection: {(navigator as any).connection?.effectiveType || 'unknown'}</div>
            <div>Online: {navigator.onLine ? 'Yes' : 'No'}</div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .debug-panel {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 9999;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .debug-panel__overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
        }

        .debug-panel__content {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: white;
          border-radius: 8px;
          width: 90vw;
          max-width: 800px;
          max-height: 90vh;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .debug-panel__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          border-bottom: 1px solid #e2e8f0;
          background: #f7fafc;
        }

        .debug-panel__header h3 {
          margin: 0;
          color: #2d3748;
        }

        .debug-panel__close {
          background: none;
          border: none;
          font-size: 20px;
          cursor: pointer;
          color: #a0aec0;
          padding: 0;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
        }

        .debug-panel__close:hover {
          background: #e2e8f0;
          color: #4a5568;
        }

        .debug-panel__section {
          padding: 20px;
          border-bottom: 1px solid #e2e8f0;
          overflow-y: auto;
        }

        .debug-panel__section:last-child {
          border-bottom: none;
        }

        .debug-panel__section h4 {
          margin: 0 0 16px 0;
          color: #2d3748;
          font-size: 16px;
        }

        .debug-panel__section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
        }

        .debug-panel__section-header h4 {
          margin: 0;
        }

        .debug-panel__actions {
          display: flex;
          gap: 8px;
        }

        .debug-panel__button {
          background: #3182ce;
          color: white;
          border: none;
          border-radius: 4px;
          padding: 8px 16px;
          font-size: 14px;
          cursor: pointer;
        }

        .debug-panel__button:hover {
          background: #2c5aa0;
        }

        .debug-panel__button:disabled {
          background: #a0aec0;
          cursor: not-allowed;
        }

        .debug-panel__button--small {
          padding: 6px 12px;
          font-size: 12px;
        }

        .debug-panel__health {
          background: #f7fafc;
          border-radius: 4px;
          padding: 16px;
        }

        .debug-panel__status {
          font-size: 16px;
          font-weight: bold;
          margin-bottom: 8px;
        }

        .debug-panel__status--healthy {
          color: #38a169;
        }

        .debug-panel__status--unhealthy {
          color: #e53e3e;
        }

        .debug-panel__health-info {
          font-size: 14px;
          color: #4a5568;
          margin-bottom: 12px;
        }

        .debug-panel__empty {
          text-align: center;
          color: #a0aec0;
          padding: 32px;
          font-style: italic;
        }

        .debug-panel__errors {
          max-height: 300px;
          overflow-y: auto;
        }

        .debug-panel__error {
          background: #fed7d7;
          border: 1px solid #feb2b2;
          border-radius: 4px;
          padding: 12px;
          margin-bottom: 12px;
        }

        .debug-panel__error-header {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-bottom: 8px;
          font-family: monospace;
          font-size: 12px;
        }

        .debug-panel__error-status {
          background: #e53e3e;
          color: white;
          padding: 2px 6px;
          border-radius: 3px;
          font-weight: bold;
        }

        .debug-panel__error-method {
          background: #2d3748;
          color: white;
          padding: 2px 6px;
          border-radius: 3px;
        }

        .debug-panel__error-endpoint {
          color: #4a5568;
          flex: 1;
          min-width: 0;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .debug-panel__copy {
          background: none;
          border: none;
          cursor: pointer;
          padding: 4px;
          border-radius: 3px;
        }

        .debug-panel__copy:hover {
          background: rgba(0, 0, 0, 0.1);
        }

        .debug-panel__error-message {
          color: #742a2a;
          margin-bottom: 8px;
          line-height: 1.4;
        }

        .debug-panel__error-meta {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 8px;
          font-size: 12px;
          color: #4a5568;
        }

        .debug-panel__error-meta code {
          background: rgba(0, 0, 0, 0.1);
          padding: 2px 4px;
          border-radius: 3px;
          font-family: monospace;
        }

        .debug-panel__env {
          background: #f7fafc;
          border-radius: 4px;
          padding: 16px;
          font-family: monospace;
          font-size: 12px;
          color: #4a5568;
        }

        .debug-panel__env > div {
          margin-bottom: 4px;
        }

        @media (max-width: 768px) {
          .debug-panel__content {
            width: 95vw;
            height: 95vh;
          }

          .debug-panel__section-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
          }

          .debug-panel__error-header {
            flex-wrap: wrap;
          }
        }
      `}</style>
    </div>
  );
};

export default DebugPanel;