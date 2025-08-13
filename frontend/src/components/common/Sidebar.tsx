import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HomeIcon,
  StarIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  XMarkIcon,
  BookOpenIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  UserIcon,
  DocumentPlusIcon,
} from '@heroicons/react/24/outline';
import {
  HomeIcon as HomeIconSolid,
  StarIcon as StarIconSolid,
  MagnifyingGlassIcon as MagnifyingGlassIconSolid,
  ChartBarIcon as ChartBarIconSolid,
  Cog6ToothIcon as Cog6ToothIconSolid,
  DocumentPlusIcon as DocumentPlusIconSolid,
} from '@heroicons/react/24/solid';
import { cn } from '@/lib/utils';

interface NavigationItem {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  activeIcon: React.ComponentType<{ className?: string }>;
  href: string;
  badge?: string;
}

interface SidebarProps {
  onClose?: () => void;
  className?: string;
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
}

const navigationItems: NavigationItem[] = [
  { id: "dashboard", name: "Dashboard", icon: HomeIcon, activeIcon: HomeIconSolid, href: "/" },
  { id: "all-papers", name: "All Papers", icon: BookOpenIcon, activeIcon: BookOpenIcon, href: "/papers" },
  { id: "collect-papers", name: "Collect Papers", icon: DocumentPlusIcon, activeIcon: DocumentPlusIconSolid, href: "/collect" },
  { id: "rate-papers", name: "Rate Papers", icon: StarIcon, activeIcon: StarIconSolid, href: "/rate" },
  { id: "search-papers", name: "Search Papers", icon: MagnifyingGlassIcon, activeIcon: MagnifyingGlassIconSolid, href: "/search" },
  { id: "analytics", name: "Analytics", icon: ChartBarIcon, activeIcon: ChartBarIconSolid, href: "/analytics" },
  { id: "settings", name: "Settings", icon: Cog6ToothIcon, activeIcon: Cog6ToothIconSolid, href: "/settings" },
];

function ModernSidebar({ onClose, className = "", collapsed, onCollapsedChange }: SidebarProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const location = useLocation();

  // Use external collapsed state if provided, otherwise use internal state
  const isCollapsed = collapsed !== undefined ? collapsed : internalCollapsed;
  
  const toggleCollapse = () => {
    const newCollapsed = !isCollapsed;
    if (onCollapsedChange) {
      onCollapsedChange(newCollapsed);
    } else {
      setInternalCollapsed(newCollapsed);
    }
  };

  const handleItemClick = () => {
    if (onClose) {
      onClose();
    }
  };

  return (
    <motion.div
      animate={{
        width: isCollapsed ? "5rem" : "18rem"
      }}
      transition={{ 
        type: "spring", 
        damping: 25, 
        stiffness: 200,
        duration: 0.3 
      }}
      className={cn(
        "flex flex-col h-full bg-white border-r border-gray-200 shadow-lg",
        className
      )}
    >
      {/* Header */}
      <motion.div 
        className="flex items-center justify-between p-6 border-b border-gray-100 bg-gray-50/50"
        layout
      >
        <AnimatePresence mode="wait">
          {!isCollapsed && (
            <motion.div
              key="expanded-header"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.2 }}
              className="flex items-center space-x-3"
            >
              <motion.div 
                className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg"
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ type: "spring", stiffness: 400 }}
              >
                <BookOpenIcon className="h-5 w-5 text-white" />
              </motion.div>
              <div className="flex flex-col">
                <span className="font-bold text-gray-900 text-lg">arXiv Hub</span>
                <span className="text-xs text-gray-500">Research Papers</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {isCollapsed && (
          <motion.div 
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center mx-auto shadow-lg"
            whileHover={{ scale: 1.1, rotate: 5 }}
          >
            <BookOpenIcon className="h-5 w-5 text-white" />
          </motion.div>
        )}

        {/* Desktop collapse button */}
        <motion.button
          onClick={toggleCollapse}
          className="hidden lg:flex p-2 rounded-lg hover:bg-gray-100 transition-all duration-200"
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
        >
          <AnimatePresence mode="wait">
            {isCollapsed ? (
              <motion.div
                key="expand"
                initial={{ rotate: -180 }}
                animate={{ rotate: 0 }}
                exit={{ rotate: 180 }}
              >
                <ChevronRightIcon className="h-4 w-4 text-gray-400" />
              </motion.div>
            ) : (
              <motion.div
                key="collapse"
                initial={{ rotate: 180 }}
                animate={{ rotate: 0 }}
                exit={{ rotate: -180 }}
              >
                <ChevronLeftIcon className="h-4 w-4 text-gray-400" />
              </motion.div>
            )}
          </AnimatePresence>
        </motion.button>

        {/* Mobile close button */}
        {onClose && (
          <button
            type="button"
            className="p-2 rounded-lg hover:bg-gray-100 lg:hidden"
            onClick={onClose}
          >
            <span className="sr-only">Close sidebar</span>
            <XMarkIcon className="h-5 w-5 text-gray-400" />
          </button>
        )}
      </motion.div>

      {/* Search Bar */}
      <AnimatePresence>
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="px-4 py-4"
          >
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search papers..."
                className="w-full pl-10 pr-4 py-2.5 bg-gray-50 border border-gray-200 rounded-xl text-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-2 overflow-y-auto">
        <motion.ul className="space-y-1" layout>
          {navigationItems.map((item, index) => {
            const Icon = item.icon;
            const ActiveIcon = item.activeIcon;
            const isActive = location.pathname === item.href;
            const IconComponent = isActive ? ActiveIcon : Icon;

            return (
              <motion.li
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <NavLink
                  to={item.href}
                  onClick={handleItemClick}
                  className={cn(
                    "w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-left transition-all duration-200 group relative overflow-hidden",
                    isActive
                      ? "bg-blue-600 text-white shadow-lg"
                      : "text-gray-600 hover:bg-gray-50 hover:text-gray-900",
                    isCollapsed ? "justify-center px-3" : ""
                  )}
                  title={isCollapsed ? item.name : undefined}
                >
                  {({ isActive: navActive }) => (
                    <>
                      {/* Background animation */}
                      {navActive && (
                        <motion.div
                          layoutId="activeBackground"
                          className="absolute inset-0 bg-blue-600 rounded-xl"
                          initial={false}
                          transition={{ type: "spring", stiffness: 400, damping: 30 }}
                        />
                      )}

                      <div className="flex items-center justify-center min-w-[24px] relative z-10">
                        <motion.div
                          whileHover={{ rotate: navActive ? 0 : 10 }}
                          transition={{ type: "spring", stiffness: 400 }}
                        >
                          <IconComponent
                            className={cn(
                              "h-5 w-5 flex-shrink-0",
                              navActive 
                                ? "text-white" 
                                : "text-gray-400 group-hover:text-gray-600"
                            )}
                          />
                        </motion.div>
                      </div>
                      
                      <AnimatePresence>
                        {!isCollapsed && (
                          <motion.div
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            className="flex items-center justify-between w-full relative z-10"
                          >
                            <span className={cn(
                              "text-sm font-medium",
                              navActive ? "text-white" : ""
                            )}>
                              {item.name}
                            </span>
                            {item.badge && (
                              <motion.span
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                className={cn(
                                  "px-2 py-1 text-xs font-bold rounded-full",
                                  navActive
                                    ? "bg-white/20 text-white"
                                    : "bg-blue-600 text-white"
                                )}
                              >
                                {item.badge}
                              </motion.span>
                            )}
                          </motion.div>
                        )}
                      </AnimatePresence>

                      {/* Badge for collapsed state */}
                      {isCollapsed && item.badge && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="absolute -top-1 -right-1 w-5 h-5 flex items-center justify-center rounded-full bg-blue-600 border-2 border-white z-20"
                        >
                          <span className="text-[10px] font-bold text-white">
                            {parseInt(item.badge) > 9 ? '9+' : item.badge}
                          </span>
                        </motion.div>
                      )}

                      {/* Tooltip for collapsed state */}
                      {isCollapsed && (
                        <div className="absolute left-full ml-3 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-50 shadow-lg">
                          {item.name}
                          {item.badge && (
                            <span className="ml-2 px-1.5 py-0.5 bg-white/20 rounded-full text-xs">
                              {item.badge}
                            </span>
                          )}
                          <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 w-2 h-2 bg-gray-900 rotate-45" />
                        </div>
                      )}
                    </>
                  )}
                </NavLink>
              </motion.li>
            );
          })}
        </motion.ul>
      </nav>

      {/* Profile Section */}
      <motion.div 
        className="mt-auto border-t border-gray-100 bg-gray-50/30"
        layout
      >
        <div className={cn("p-4", isCollapsed ? "px-3" : "")}>
          <AnimatePresence mode="wait">
            {!isCollapsed ? (
              <motion.div
                key="expanded-profile"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="flex items-center px-3 py-3 rounded-xl bg-white hover:bg-gray-50 transition-colors duration-200 cursor-pointer"
                whileHover={{ scale: 1.02 }}
              >
                <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                  <UserIcon className="h-5 w-5 text-gray-400" />
                </div>
                <div className="flex-1 min-w-0 ml-3">
                  <p className="text-sm font-medium text-gray-900 truncate">Research User</p>
                  <p className="text-xs text-gray-500 truncate">Academic Researcher</p>
                </div>
                <div className="w-3 h-3 bg-green-500 rounded-full ml-2" title="Online" />
              </motion.div>
            ) : (
              <motion.div
                key="collapsed-profile"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                className="flex justify-center"
              >
                <div className="relative">
                  <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                    <UserIcon className="h-5 w-5 text-gray-400" />
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white" />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </motion.div>
  );
}

export const Sidebar: React.FC<SidebarProps> = ({ onClose, className, collapsed, onCollapsedChange }) => {
  return <ModernSidebar onClose={onClose} className={className} collapsed={collapsed} onCollapsedChange={onCollapsedChange} />;
};