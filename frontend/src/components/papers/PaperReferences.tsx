import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  DocumentTextIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  LinkIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  ArrowTopRightOnSquareIcon,
  BookOpenIcon
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import toast from 'react-hot-toast';

import { LoadingSpinner } from '@components/common/LoadingSpinner';
import { GraphVisualization } from '@components/graph/GraphVisualization';

interface Reference {
  id: number;
  cited_paper_id?: string;
  cited_title?: string;
  cited_authors?: string;
  cited_year?: number;
  reference_context: string;
  citation_number?: number;
  is_arxiv_paper: boolean;
  cited_paper_url?: string;
  created_at: string;
}

interface Citation {
  citing_paper_id: string;
  citing_title: string;
  citing_authors: string;
  cited_context?: string;
  citation_count: number;
}

interface NetworkNode {
  id: string;
  title: string;
  type: 'source' | 'reference' | 'citation';
  arxiv_id?: string;
  year?: number;
  authors?: string;
}

interface PaperReferencesProps {
  paperId: string;
  paperTitle: string;
  className?: string;
}

export const PaperReferences: React.FC<PaperReferencesProps> = ({
  paperId,
  paperTitle: _paperTitle,
  className
}) => {
  const [references, setReferences] = useState<Reference[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [networkData, setNetworkData] = useState<NetworkNode[]>([]);
  const [loading, setLoading] = useState({
    references: false,
    citations: false,
    fetching: false,
    network: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'references' | 'citations' | 'network' | 'graph'>('references');
  const [isExpanded, setIsExpanded] = useState(false);

  // Fetch references on component mount
  useEffect(() => {
    fetchReferences();
    fetchCitations();
  }, [paperId]);

  // Additional fetch when expanded (in case of refresh needs)
  useEffect(() => {
    if (isExpanded && references.length === 0) {
      fetchReferences();
    }
  }, [isExpanded]);

  const fetchReferences = async () => {
    try {
      setLoading(prev => ({ ...prev, references: true }));
      setError(null);
      
      const response = await fetch(`/api/papers/${paperId}/references`);
      if (!response.ok) {
        throw new Error('Failed to fetch references');
      }
      
      const data = await response.json();
      setReferences(data.data?.references || []);
    } catch (err: any) {
      console.error('Error fetching references:', err);
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, references: false }));
    }
  };

  const fetchCitations = async () => {
    try {
      setLoading(prev => ({ ...prev, citations: true }));
      
      const response = await fetch(`/api/papers/${paperId}/citations`);
      if (!response.ok) {
        throw new Error('Failed to fetch citations');
      }
      
      const data = await response.json();
      setCitations(data.data?.citations || []);
    } catch (err: any) {
      console.error('Error fetching citations:', err);
      // Don't set error for citations failure, as it's less critical
    } finally {
      setLoading(prev => ({ ...prev, citations: false }));
    }
  };

  const fetchPaperReferences = async () => {
    try {
      setLoading(prev => ({ ...prev, fetching: true }));
      setError(null);
      
      const response = await fetch(`/api/references/fetch/${paperId}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch references from ArXiv');
      }
      
      const data = await response.json();
      const referencesFound = data.data?.references_found || 0;
      const referencesEnhanced = data.data?.references_enhanced || 0;
      const source = data.data?.source || 'grobid';
      
      if (referencesEnhanced > 0) {
        toast.success(`Fetched ${referencesFound} references (${referencesEnhanced} AI-validated and enhanced via DuckDuckGo)`);
      } else {
        toast.success(`Fetched ${referencesFound} references from ${source}`);
      }
      
      // Refresh references after fetching
      await fetchReferences();
    } catch (err: any) {
      console.error('Error fetching paper references:', err);
      toast.error(err.message || 'Failed to fetch references');
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, fetching: false }));
    }
  };

  const fetchCitationNetwork = async () => {
    try {
      setLoading(prev => ({ ...prev, network: true }));
      
      const response = await fetch(`/api/references/network/${paperId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch citation network');
      }
      
      const data = await response.json();
      setNetworkData(data.data?.nodes || []);
    } catch (err: any) {
      console.error('Error fetching citation network:', err);
      toast.error('Failed to load citation network');
    } finally {
      setLoading(prev => ({ ...prev, network: false }));
    }
  };

  const handleTabChange = (tab: 'references' | 'citations' | 'network' | 'graph') => {
    setActiveTab(tab);
    if (tab === 'network' && networkData.length === 0) {
      fetchCitationNetwork();
    }
  };

  const formatReferenceDisplay = (ref: Reference): string => {
    const parts: string[] = [];
    
    if (ref.cited_authors) {
      parts.push(ref.cited_authors);
    }
    
    if (ref.cited_title) {
      parts.push(`"${ref.cited_title}"`);
    }
    
    if (ref.cited_year) {
      parts.push(`(${ref.cited_year})`);
    }
    
    return parts.join(', ') || ref.reference_context.substring(0, 100) + '...';
  };

  const generateGoogleSearchUrl = (ref: Reference): string => {
    const searchTerms: string[] = [];
    
    if (ref.cited_title) {
      searchTerms.push(`"${ref.cited_title}"`);
    }
    
    if (ref.cited_authors) {
      // Extract first author name for search
      const firstAuthor = ref.cited_authors.split(',')[0].trim();
      searchTerms.push(firstAuthor);
    }
    
    if (ref.cited_year) {
      searchTerms.push(ref.cited_year.toString());
    }
    
    const query = searchTerms.join(' ');
    return `https://www.google.com/search?q=${encodeURIComponent(query)}`;
  };

  const ReferenceLink: React.FC<{ reference: Reference; children: React.ReactNode }> = ({ 
    reference, 
    children 
  }) => {
    // If we have a cited_paper_id and it exists in our database, link internally
    if (reference.cited_paper_id && reference.cited_paper_url) {
      return (
        <Link
          to={`/papers/${reference.cited_paper_id}`}
          className="block text-gray-900 hover:text-primary-600 transition-colors"
        >
          {children}
        </Link>
      );
    }
    
    // If we have citation info but no internal link, provide Google search
    if (reference.cited_title || reference.cited_authors) {
      return (
        <a
          href={generateGoogleSearchUrl(reference)}
          target="_blank"
          rel="noopener noreferrer"
          className="block text-gray-900 hover:text-primary-600 transition-colors"
          title="Search on Google"
        >
          {children}
        </a>
      );
    }
    
    // No actionable link available
    return <div className="block text-gray-900">{children}</div>;
  };

  const totalReferencesCount = references.length;
  const arxivReferencesCount = references.filter(ref => ref.is_arxiv_paper).length;
  const totalCitationsCount = citations.length;

  return (
    <div className={clsx('bg-white border border-gray-200 rounded-lg', className)}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center space-x-3 hover:bg-gray-50 -m-2 p-2 rounded-lg transition-colors"
          >
            <div className="flex items-center space-x-2">
              {isExpanded ? (
                <ChevronDownIcon className="h-4 w-4 text-gray-500" />
              ) : (
                <ChevronRightIcon className="h-4 w-4 text-gray-500" />
              )}
              <BookOpenIcon className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-900">
                References & Citations
              </h3>
              <p className="text-xs text-gray-500">
                {totalReferencesCount > 0 && `${totalReferencesCount} references`}
                {totalReferencesCount > 0 && totalCitationsCount > 0 && ' • '}
                {totalCitationsCount > 0 && `${totalCitationsCount} citations`}
                {totalReferencesCount === 0 && totalCitationsCount === 0 && 'View citation data'}
              </p>
            </div>
          </button>
          
          <div className="flex items-center space-x-2">
            {arxivReferencesCount > 0 && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                {arxivReferencesCount} ArXiv
              </span>
            )}
            {totalReferencesCount === 0 ? (
              <button
                onClick={fetchPaperReferences}
                disabled={loading.fetching}
                className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 hover:bg-blue-200 transition-colors disabled:opacity-50"
              >
                {loading.fetching ? (
                  <>
                    <LoadingSpinner size="xs" className="mr-1" />
                    Fetching...
                  </>
                ) : (
                  <>
                    <MagnifyingGlassIcon className="h-3 w-3 mr-1" />
                    Fetch References
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={fetchPaperReferences}
                disabled={loading.fetching}
                className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors disabled:opacity-50"
                title="Re-fetch references from ArXiv"
              >
                {loading.fetching ? (
                  <>
                    <LoadingSpinner size="xs" className="mr-1" />
                    Re-fetching...
                  </>
                ) : (
                  <>
                    <MagnifyingGlassIcon className="h-3 w-3 mr-1" />
                    Re-fetch
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {/* Tab Navigation */}
            <div className="border-b border-gray-200">
              <nav className="flex space-x-8 px-4" aria-label="Tabs">
                {[
                  { key: 'references', label: 'References', count: totalReferencesCount },
                  { key: 'citations', label: 'Citations', count: totalCitationsCount },
                  { key: 'network', label: 'Network', count: networkData.length },
                  { key: 'graph', label: 'Graph', count: 0 },
                ].map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => handleTabChange(tab.key as any)}
                    className={clsx(
                      'py-2 px-1 border-b-2 font-medium text-sm transition-colors',
                      activeTab === tab.key
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    )}
                  >
                    {tab.label}
                    {tab.count > 0 && (
                      <span className={clsx(
                        'ml-2 py-0.5 px-2 rounded-full text-xs',
                        activeTab === tab.key
                          ? 'bg-blue-100 text-blue-600'
                          : 'bg-gray-100 text-gray-900'
                      )}>
                        {tab.count}
                      </span>
                    )}
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab Content */}
            <div className="p-4">
              {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <ExclamationTriangleIcon className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-red-700">{error}</p>
                  </div>
                </div>
              )}

              {/* References Tab */}
              {activeTab === 'references' && (
                <div className="space-y-4">
                  {loading.references ? (
                    <div className="flex items-center justify-center py-8">
                      <LoadingSpinner size="sm" />
                      <span className="ml-2 text-sm text-gray-500">Loading references...</span>
                    </div>
                  ) : references.length > 0 ? (
                    <div className="space-y-3">
                      {references.map((ref, index) => (
                        <motion.div
                          key={ref.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className="p-3 bg-gray-50 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <span className="text-xs font-medium text-gray-500">
                                  #{ref.citation_number || index + 1}
                                </span>
                                
                                {/* Status indicators */}
                                {ref.cited_paper_id && ref.cited_paper_url && (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                                    <LinkIcon className="w-3 h-3 mr-1" />
                                    In Database
                                  </span>
                                )}
                                
                                {Boolean(ref.is_arxiv_paper) && (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                    ArXiv
                                  </span>
                                )}
                                
                                {ref.cited_year && (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                    {ref.cited_year}
                                  </span>
                                )}
                                
                                {/* Clickable indicator */}
                                {(ref.cited_title || ref.cited_authors) && (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                                    <MagnifyingGlassIcon className="w-3 h-3 mr-1" />
                                    Searchable
                                  </span>
                                )}
                              </div>
                              
                              <ReferenceLink reference={ref}>
                                <div className="cursor-pointer">
                                  <p className="text-sm font-medium mb-2 hover:underline">
                                    {formatReferenceDisplay(ref)}
                                  </p>
                                  
                                  {ref.reference_context && ref.reference_context !== formatReferenceDisplay(ref) && (
                                    <p className="text-xs text-gray-600 bg-white p-2 rounded border">
                                      {ref.reference_context.length > 200 
                                        ? ref.reference_context.substring(0, 200) + '...'
                                        : ref.reference_context
                                      }
                                    </p>
                                  )}
                                </div>
                              </ReferenceLink>
                            </div>
                            
                            <div className="flex items-center space-x-1 ml-3">
                              {/* Internal link indicator */}
                              {ref.cited_paper_id && ref.cited_paper_url ? (
                                <span 
                                  className="p-1 text-primary-500 hover:text-primary-600"
                                  title="Available in database - click to view"
                                >
                                  <DocumentTextIcon className="h-4 w-4" />
                                </span>
                              ) : (ref.cited_title || ref.cited_authors) ? (
                                <span 
                                  className="p-1 text-gray-400 hover:text-gray-600"
                                  title="Click to search on Google"
                                >
                                  <MagnifyingGlassIcon className="h-4 w-4" />
                                </span>
                              ) : null}
                              
                              {/* External ArXiv link if available */}
                              {ref.cited_paper_id && Boolean(ref.is_arxiv_paper) && (
                                <a
                                  href={`https://arxiv.org/abs/${ref.cited_paper_id}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="p-1 text-gray-400 hover:text-blue-600 transition-colors"
                                  title="View on ArXiv"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                                </a>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
                      <h3 className="mt-2 text-sm font-medium text-gray-900">No references found</h3>
                      <p className="mt-1 text-sm text-gray-500">
                        This paper's references haven't been parsed yet.
                      </p>
                      <div className="mt-4">
                        <button
                          onClick={fetchPaperReferences}
                          disabled={loading.fetching}
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {loading.fetching ? (
                            <>
                              <LoadingSpinner size="sm" className="mr-2" />
                              Fetching from ArXiv...
                            </>
                          ) : (
                            <>
                              <MagnifyingGlassIcon className="h-4 w-4 mr-2" />
                              Fetch References
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Citations Tab */}
              {activeTab === 'citations' && (
                <div className="space-y-4">
                  {loading.citations ? (
                    <div className="flex items-center justify-center py-8">
                      <LoadingSpinner size="sm" />
                      <span className="ml-2 text-sm text-gray-500">Loading citations...</span>
                    </div>
                  ) : citations.length > 0 ? (
                    <div className="space-y-3">
                      {citations.map((citation, index) => (
                        <motion.div
                          key={citation.citing_paper_id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className="p-3 bg-gray-50 rounded-lg border border-gray-200"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className="text-sm font-medium text-gray-900 mb-1">
                                {citation.citing_title}
                              </h4>
                              <p className="text-xs text-gray-600 mb-2">
                                by {citation.citing_authors}
                              </p>
                              {citation.cited_context && (
                                <p className="text-xs text-gray-600 bg-white p-2 rounded border">
                                  {citation.cited_context}
                                </p>
                              )}
                            </div>
                            
                            <a
                              href={`https://arxiv.org/abs/${citation.citing_paper_id}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="ml-3 p-1 text-gray-400 hover:text-blue-600 transition-colors"
                              title="View citing paper"
                            >
                              <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                            </a>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <LinkIcon className="mx-auto h-12 w-12 text-gray-400" />
                      <h3 className="mt-2 text-sm font-medium text-gray-900">No citations found</h3>
                      <p className="mt-1 text-sm text-gray-500">
                        No papers have been found that cite this work yet.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Network Tab */}
              {activeTab === 'network' && (
                <div className="space-y-4">
                  {loading.network ? (
                    <div className="flex items-center justify-center py-8">
                      <LoadingSpinner size="sm" />
                      <span className="ml-2 text-sm text-gray-500">Loading citation network...</span>
                    </div>
                  ) : networkData.length > 0 ? (
                    <div className="space-y-4">
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <p className="text-sm text-blue-700">
                          <strong>Citation Network:</strong> Visual representation of how this paper connects 
                          to other research through references and citations.
                        </p>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {networkData.map((node, index) => (
                          <motion.div
                            key={node.id}
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.1 }}
                            className={clsx(
                              'p-3 rounded-lg border',
                              node.type === 'source' && 'bg-yellow-50 border-yellow-200',
                              node.type === 'reference' && 'bg-green-50 border-green-200',
                              node.type === 'citation' && 'bg-blue-50 border-blue-200'
                            )}
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <span className={clsx(
                                  'inline-block px-2 py-1 rounded-full text-xs font-medium mb-2',
                                  node.type === 'source' && 'bg-yellow-100 text-yellow-800',
                                  node.type === 'reference' && 'bg-green-100 text-green-800',
                                  node.type === 'citation' && 'bg-blue-100 text-blue-800'
                                )}>
                                  {node.type}
                                </span>
                                <p className="text-sm font-medium text-gray-900 mb-1">
                                  {node.title.length > 60 ? node.title.substring(0, 60) + '...' : node.title}
                                </p>
                                {node.authors && (
                                  <p className="text-xs text-gray-600">
                                    {node.authors.length > 40 ? node.authors.substring(0, 40) + '...' : node.authors}
                                  </p>
                                )}
                                {node.year && (
                                  <p className="text-xs text-gray-500 mt-1">{node.year}</p>
                                )}
                              </div>
                              
                              {node.arxiv_id && (
                                <a
                                  href={`https://arxiv.org/abs/${node.arxiv_id}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="ml-2 p-1 text-gray-400 hover:text-blue-600 transition-colors"
                                >
                                  <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                                </a>
                              )}
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="mx-auto h-12 w-12 text-gray-400">
                        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                        </svg>
                      </div>
                      <h3 className="mt-2 text-sm font-medium text-gray-900">No network data</h3>
                      <p className="mt-1 text-sm text-gray-500">
                        Citation network data is not available for this paper.
                      </p>
                      <div className="mt-4">
                        <button
                          onClick={fetchCitationNetwork}
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-blue-600 bg-blue-100 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Refresh Network Data
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Graph Tab */}
              {activeTab === 'graph' && (
                <div className="space-y-4">
                  <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                    <p className="text-sm text-purple-700">
                      <strong>Citation Graph:</strong> Interactive graph visualization of citation networks 
                      using graph database analysis. Shows papers as nodes and citations as directed edges.
                    </p>
                  </div>
                  
                  <GraphVisualization 
                    paperId={paperId}
                    className="border-0"
                  />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};