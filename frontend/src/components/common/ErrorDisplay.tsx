import React, { useState, useEffect } from 'react';
import { ApiError } from '../../types/api';

interface ErrorDisplayProps {
  error?: ApiError | null;
  onDismiss?: () => void;
  showDetails?: boolean;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ 
  error, 
  onDismiss, 
  showDetails = false 
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [showFullDetails, setShowFullDetails] = useState(false);

  useEffect(() => {
    if (error) {
      setIsVisible(true);
    }
  }, [error]);

  const handleDismiss = () => {
    setIsVisible(false);
    setTimeout(() => {
      onDismiss?.();
    }, 300); // Wait for animation
  };

  if (!error || !isVisible) return null;

  const getErrorIcon = (statusCode: number) => {
    if (statusCode >= 500) return '🚨';
    if (statusCode >= 400) return '⚠️';
    return '❌';
  };

  const getErrorSeverity = (statusCode: number) => {
    if (statusCode >= 500) return 'error';
    if (statusCode >= 400) return 'warning';
    return 'info';
  };

  const severity = getErrorSeverity(error.statusCode);
  const icon = getErrorIcon(error.statusCode);

  return (
    <div className={`error-display error-display--${severity} ${isVisible ? 'error-display--visible' : ''}`}>
      <div className="error-display__header">
        <span className="error-display__icon">{icon}</span>
        <div className="error-display__title">
          <h4>Request Failed ({error.statusCode})</h4>
          {error.correlationId && (
            <small className="error-display__correlation">
              ID: {error.correlationId}
            </small>
          )}
        </div>
        <button 
          className="error-display__close" 
          onClick={handleDismiss}
          aria-label="Dismiss error"
        >
          ✕
        </button>
      </div>

      <div className="error-display__message">
        {error.message}
      </div>

      {showDetails && (
        <div className="error-display__details">
          <button 
            className="error-display__toggle"
            onClick={() => setShowFullDetails(!showFullDetails)}
          >
            {showFullDetails ? 'Hide' : 'Show'} Details
          </button>
          
          {showFullDetails && (
            <div className="error-display__debug">
              <div className="error-display__debug-grid">
                <div>
                  <strong>Method:</strong> {error.method || 'Unknown'}
                </div>
                <div>
                  <strong>Endpoint:</strong> {error.endpoint || 'Unknown'}
                </div>
                <div>
                  <strong>Status:</strong> {error.statusCode}
                </div>
                <div>
                  <strong>Time:</strong> {error.responseTime || 'Unknown'}
                </div>
                <div>
                  <strong>Timestamp:</strong> {new Date(error.timestamp).toLocaleString()}
                </div>
                {error.correlationId && (
                  <div>
                    <strong>Correlation ID:</strong> 
                    <code className="error-display__code">{error.correlationId}</code>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      <style jsx>{`
        .error-display {
          position: fixed;
          top: 20px;
          right: 20px;
          max-width: 400px;
          background: white;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          border-left: 4px solid #e53e3e;
          opacity: 0;
          transform: translateX(100%);
          transition: all 0.3s ease-in-out;
          z-index: 1000;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .error-display--visible {
          opacity: 1;
          transform: translateX(0);
        }

        .error-display--error {
          border-left-color: #e53e3e;
        }

        .error-display--warning {
          border-left-color: #d69e2e;
        }

        .error-display--info {
          border-left-color: #3182ce;
        }

        .error-display__header {
          display: flex;
          align-items: flex-start;
          padding: 16px 16px 8px;
          gap: 12px;
        }

        .error-display__icon {
          font-size: 20px;
          flex-shrink: 0;
        }

        .error-display__title {
          flex: 1;
          min-width: 0;
        }

        .error-display__title h4 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
          color: #1a202c;
        }

        .error-display__correlation {
          color: #718096;
          font-size: 12px;
          font-family: monospace;
        }

        .error-display__close {
          background: none;
          border: none;
          font-size: 18px;
          cursor: pointer;
          color: #a0aec0;
          flex-shrink: 0;
          padding: 0;
          width: 24px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
        }

        .error-display__close:hover {
          background: #f7fafc;
          color: #4a5568;
        }

        .error-display__message {
          padding: 0 16px 16px;
          color: #4a5568;
          line-height: 1.5;
        }

        .error-display__details {
          border-top: 1px solid #e2e8f0;
          padding: 12px 16px;
        }

        .error-display__toggle {
          background: #f7fafc;
          border: 1px solid #e2e8f0;
          border-radius: 4px;
          padding: 6px 12px;
          font-size: 12px;
          cursor: pointer;
          color: #4a5568;
        }

        .error-display__toggle:hover {
          background: #edf2f7;
        }

        .error-display__debug {
          margin-top: 12px;
          background: #f7fafc;
          border-radius: 4px;
          padding: 12px;
        }

        .error-display__debug-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 8px;
          font-size: 12px;
        }

        .error-display__debug-grid > div {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .error-display__debug-grid strong {
          color: #2d3748;
          margin-right: 8px;
        }

        .error-display__code {
          background: #e2e8f0;
          padding: 2px 6px;
          border-radius: 3px;
          font-family: monospace;
          font-size: 11px;
          color: #4a5568;
        }

        @media (max-width: 480px) {
          .error-display {
            top: 10px;
            right: 10px;
            left: 10px;
            max-width: none;
          }
        }
      `}</style>
    </div>
  );
};

export default ErrorDisplay;