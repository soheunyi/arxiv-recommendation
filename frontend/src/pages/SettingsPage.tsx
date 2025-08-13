import React from 'react';

export const SettingsPage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-secondary-900">⚙️ Settings</h1>
        <p className="mt-2 text-secondary-600">
          Configure your preferences and system settings.
        </p>
      </div>
      
      {/* Coming soon placeholder */}
      <div className="text-center py-12">
        <div className="text-6xl mb-4">⚙️</div>
        <h3 className="text-lg font-medium text-secondary-900 mb-2">
          Settings Panel Coming Soon
        </h3>
        <p className="text-secondary-600">
          This page will allow you to customize your experience and manage system configuration.
        </p>
      </div>
    </div>
  );
};