import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bars3Icon, 
  BellIcon,
  Cog6ToothIcon 
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

import { Sidebar } from './Sidebar';
import { useAppSelector } from '@store';

export const Layout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isDesktop, setIsDesktop] = useState(false);
  const { isOnline, notifications } = useAppSelector((state) => state.system);
  const unreadCount = notifications.filter(n => !n.read).length;
  
  // Handle window resize for responsive behavior
  useEffect(() => {
    const checkIsDesktop = () => {
      setIsDesktop(window.innerWidth >= 1024);
    };
    
    checkIsDesktop();
    window.addEventListener('resize', checkIsDesktop);
    
    return () => window.removeEventListener('resize', checkIsDesktop);
  }, []);
  
  return (
    <div className="min-h-screen bg-secondary-50">
      {/* Mobile sidebar backdrop */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <div className="absolute inset-0 bg-secondary-600 opacity-75" />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:flex-col lg:z-50">
        <Sidebar 
          collapsed={sidebarCollapsed}
          onCollapsedChange={setSidebarCollapsed}
        />
      </div>
      
      {/* Mobile sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed inset-y-0 left-0 z-50 w-64 lg:hidden"
          >
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Main content */}
      <motion.div 
        className="flex flex-col"
        animate={{
          paddingLeft: isDesktop ? (sidebarCollapsed ? "5rem" : "18rem") : "0rem"
        }}
        transition={{ 
          type: "spring", 
          damping: 25, 
          stiffness: 200,
          duration: 0.3 
        }}
      >
        {/* Top header */}
        <header className="sticky top-0 z-30 flex h-16 items-center gap-x-4 border-b border-secondary-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
          {/* Mobile menu button */}
          <button
            type="button"
            className="-m-2.5 p-2.5 text-secondary-700 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <span className="sr-only">Open sidebar</span>
            <Bars3Icon className="h-6 w-6" aria-hidden="true" />
          </button>
          
          {/* Separator */}
          <div className="h-6 w-px bg-secondary-200 lg:hidden" aria-hidden="true" />
          
          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            {/* Title area */}
            <div className="flex items-center">
              <h1 className="text-lg font-semibold text-secondary-900">
                arXiv Recommendations
              </h1>
            </div>
            
            <div className="flex items-center gap-x-4 lg:gap-x-6 ml-auto">
              {/* Online status indicator */}
              <div className="flex items-center gap-2">
                <div className={clsx(
                  'w-2 h-2 rounded-full',
                  isOnline ? 'bg-green-500' : 'bg-red-500'
                )} />
                <span className="text-sm text-secondary-600">
                  {isOnline ? 'Online' : 'Offline'}
                </span>
              </div>
              
              {/* Notifications */}
              <button
                type="button"
                className="relative -m-2.5 p-2.5 text-secondary-400 hover:text-secondary-500"
              >
                <span className="sr-only">View notifications</span>
                <BellIcon className="h-6 w-6" aria-hidden="true" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white">
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </span>
                )}
              </button>
              
              {/* Settings */}
              <button
                type="button"
                className="-m-2.5 p-2.5 text-secondary-400 hover:text-secondary-500"
              >
                <span className="sr-only">Settings</span>
                <Cog6ToothIcon className="h-6 w-6" aria-hidden="true" />
              </button>
            </div>
          </div>
        </header>
        
        {/* Page content */}
        <main className="flex-1">
          <div className={`px-4 py-6 sm:px-6 ${sidebarCollapsed ? 'lg:px-12' : 'lg:px-8'} transition-all duration-300`}>
            <Outlet />
          </div>
        </main>
      </motion.div>
    </div>
  );
};