import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  EyeIcon,
  ArrowPathIcon,
  AdjustmentsHorizontalIcon,
  InformationCircleIcon,
  MagnifyingGlassIcon,
  ArrowTopRightOnSquareIcon
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import toast from 'react-hot-toast';

import { LoadingSpinner } from '@components/common/LoadingSpinner';

interface GraphNode {
  id: string;
  level: number;
  properties: {
    title: string;
    authors: string[];
    year: number;
    categories: string[];
  };
}

interface GraphEdge {
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
  weight: number;
}

interface GraphData {
  center_paper: string;
  depth: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
}

interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_counts: Record<string, number>;
  edge_counts: Record<string, number>;
  avg_degree: number;
  density: number;
  computed_at: string;
}

interface GraphVisualizationProps {
  paperId: string;
  className?: string;
}

export const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  paperId,
  className
}) => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [loading, setLoading] = useState({
    subgraph: false,
    stats: false,
    initialize: false,
    sync: false
  });
  const [error, setError] = useState<string | null>(null);
  const [depth, setDepth] = useState(2);
  const [isInitialized, setIsInitialized] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

  // Check if graph database is initialized
  useEffect(() => {
    checkGraphInitialization();
  }, []);

  // Load subgraph when paperId or depth changes
  useEffect(() => {
    if (isInitialized && paperId) {
      loadSubgraph();
    }
  }, [paperId, depth, isInitialized]);

  const checkGraphInitialization = async () => {
    try {
      const response = await fetch('/api/graph/stats');
      if (response.ok) {
        const stats = await response.json();
        setGraphStats(stats);
        setIsInitialized(stats.total_nodes > 0);
      } else {
        setIsInitialized(false);
      }
    } catch (err) {
      console.error('Error checking graph initialization:', err);
      setIsInitialized(false);
    }
  };

  const initializeGraphDatabase = async () => {
    try {
      setLoading(prev => ({ ...prev, initialize: true }));
      setError(null);
      
      const response = await fetch('/api/graph/initialize', {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to initialize graph database');
      }
      
      const result = await response.json();
      toast.success(`Graph initialized: ${result.sync_result.papers_added} papers, ${result.sync_result.edges_added} edges`);
      
      setIsInitialized(true);
      await checkGraphInitialization();
      
    } catch (err: any) {
      console.error('Error initializing graph database:', err);
      setError(err.message);
      toast.error('Failed to initialize graph database');
    } finally {
      setLoading(prev => ({ ...prev, initialize: false }));
    }
  };

  const syncGraphData = async () => {
    try {
      setLoading(prev => ({ ...prev, sync: true }));
      setError(null);
      
      const response = await fetch('/api/graph/sync', {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error('Failed to sync graph data');
      }
      
      const result = await response.json();
      toast.success(`Graph synced: ${result.sync_result.papers_added} papers, ${result.sync_result.edges_added} edges`);
      
      await checkGraphInitialization();
      if (paperId) {
        await loadSubgraph();
      }
      
    } catch (err: any) {
      console.error('Error syncing graph data:', err);
      setError(err.message);
      toast.error('Failed to sync graph data');
    } finally {
      setLoading(prev => ({ ...prev, sync: false }));
    }
  };

  const loadSubgraph = async () => {
    try {
      setLoading(prev => ({ ...prev, subgraph: true }));
      setError(null);
      
      const response = await fetch(`/api/graph/subgraph/${paperId}?depth=${depth}`);
      if (!response.ok) {
        throw new Error('Failed to load citation subgraph');
      }
      
      const data = await response.json();
      setGraphData(data.subgraph);
      
      // Render graph visualization
      renderGraph(data.subgraph);
      
    } catch (err: any) {
      console.error('Error loading subgraph:', err);
      setError(err.message);
    } finally {
      setLoading(prev => ({ ...prev, subgraph: false }));
    }
  };

  const loadGraphStats = async () => {
    try {
      setLoading(prev => ({ ...prev, stats: true }));
      
      const response = await fetch('/api/graph/stats');
      if (!response.ok) {
        throw new Error('Failed to load graph statistics');
      }
      
      const stats = await response.json();
      setGraphStats(stats);
      
    } catch (err: any) {
      console.error('Error loading graph stats:', err);
      toast.error('Failed to load graph statistics');
    } finally {
      setLoading(prev => ({ ...prev, stats: false }));
    }
  };

  const renderGraph = (data: GraphData) => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = svgRef.current;
    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;

    // Clear previous content
    svg.innerHTML = '';

    // Create simple circular layout
    const nodePositions = new Map<string, { x: number; y: number }>();
    
    // Position center node
    const centerNode = data.nodes.find(n => n.id === data.center_paper);
    if (centerNode) {
      nodePositions.set(centerNode.id, { x: centerX, y: centerY });
    }

    // Position nodes by level in concentric circles
    const levelGroups = new Map<number, GraphNode[]>();
    data.nodes.forEach(node => {
      if (node.id !== data.center_paper) {
        const level = node.level || 1;
        if (!levelGroups.has(level)) {
          levelGroups.set(level, []);
        }
        levelGroups.get(level)!.push(node);
      }
    });

    levelGroups.forEach((nodes, level) => {
      const radius = level * 120;
      const angleStep = (2 * Math.PI) / nodes.length;
      
      nodes.forEach((node, index) => {
        const angle = index * angleStep;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        nodePositions.set(node.id, { x, y });
      });
    });

    // Create SVG groups
    const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    edgeGroup.setAttribute('class', 'edges');
    svg.appendChild(edgeGroup);

    const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    nodeGroup.setAttribute('class', 'nodes');
    svg.appendChild(nodeGroup);

    // Draw edges
    data.edges.forEach(edge => {
      const sourcePos = nodePositions.get(edge.source);
      const targetPos = nodePositions.get(edge.target);
      
      if (sourcePos && targetPos) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', sourcePos.x.toString());
        line.setAttribute('y1', sourcePos.y.toString());
        line.setAttribute('x2', targetPos.x.toString());
        line.setAttribute('y2', targetPos.y.toString());
        line.setAttribute('stroke', '#cbd5e1');
        line.setAttribute('stroke-width', '1');
        line.setAttribute('marker-end', 'url(#arrowhead)');
        edgeGroup.appendChild(line);
      }
    });

    // Add arrow marker definition
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'arrowhead');
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '7');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '3.5');
    marker.setAttribute('orient', 'auto');
    
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
    polygon.setAttribute('fill', '#cbd5e1');
    marker.appendChild(polygon);
    defs.appendChild(marker);
    svg.insertBefore(defs, edgeGroup);

    // Draw nodes
    data.nodes.forEach(node => {
      const pos = nodePositions.get(node.id);
      if (!pos) return;

      const isCenter = node.id === data.center_paper;
      const nodeRadius = isCenter ? 20 : 15;
      const fillColor = isCenter ? '#3b82f6' : '#10b981';

      // Node circle
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pos.x.toString());
      circle.setAttribute('cy', pos.y.toString());
      circle.setAttribute('r', nodeRadius.toString());
      circle.setAttribute('fill', fillColor);
      circle.setAttribute('stroke', '#ffffff');
      circle.setAttribute('stroke-width', '2');
      circle.setAttribute('cursor', 'pointer');
      
      // Add click handler
      circle.addEventListener('click', () => {
        // Navigate to paper or show details
        window.open(`/papers/${node.id}`, '_blank');
      });
      
      nodeGroup.appendChild(circle);

      // Node label (truncated title)
      if (node.properties.title) {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        const truncatedTitle = node.properties.title.length > 30 
          ? node.properties.title.substring(0, 30) + '...'
          : node.properties.title;
        
        text.textContent = truncatedTitle;
        text.setAttribute('x', pos.x.toString());
        text.setAttribute('y', (pos.y + nodeRadius + 15).toString());
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '10');
        text.setAttribute('fill', '#374151');
        text.setAttribute('cursor', 'pointer');
        
        text.addEventListener('click', () => {
          window.open(`/papers/${node.id}`, '_blank');
        });
        
        nodeGroup.appendChild(text);
      }
    });
  };

  if (!isInitialized) {
    return (
      <div className={clsx('bg-white border border-gray-200 rounded-lg p-6', className)}>
        <div className="text-center">
          <div className="mx-auto h-12 w-12 text-gray-400 mb-4">
            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Graph Database Not Initialized</h3>
          <p className="text-sm text-gray-600 mb-4">
            Initialize the graph database to enable citation network visualization.
          </p>
          <button
            onClick={initializeGraphDatabase}
            disabled={loading.initialize}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {loading.initialize ? (
              <>
                <LoadingSpinner size="sm" className="mr-2" />
                Initializing...
              </>
            ) : (
              <>
                <ArrowPathIcon className="h-4 w-4 mr-2" />
                Initialize Graph Database
              </>
            )}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('bg-white border border-gray-200 rounded-lg', className)}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <EyeIcon className="h-5 w-5 text-purple-600" />
              <h3 className="text-sm font-semibold text-gray-900">Citation Graph</h3>
            </div>
            
            {graphData && (
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                {graphData.node_count} nodes, {graphData.edge_count} edges
              </span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Depth Control */}
            <div className="flex items-center space-x-2">
              <label className="text-xs text-gray-500">Depth:</label>
              <select
                value={depth}
                onChange={(e) => setDepth(parseInt(e.target.value))}
                className="text-xs border border-gray-300 rounded px-2 py-1"
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </div>
            
            {/* Stats Toggle */}
            <button
              onClick={() => {
                setShowStats(!showStats);
                if (!showStats && !graphStats) {
                  loadGraphStats();
                }
              }}
              className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-700 hover:bg-gray-200"
            >
              <InformationCircleIcon className="h-3 w-3 mr-1" />
              Stats
            </button>
            
            {/* Sync Button */}
            <button
              onClick={syncGraphData}
              disabled={loading.sync}
              className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-700 hover:bg-blue-200 disabled:opacity-50"
            >
              {loading.sync ? (
                <LoadingSpinner size="xs" className="mr-1" />
              ) : (
                <ArrowPathIcon className="h-3 w-3 mr-1" />
              )}
              Sync
            </button>
          </div>
        </div>
      </div>

      {/* Stats Panel */}
      {showStats && graphStats && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="p-4 bg-gray-50 border-b border-gray-200"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="font-medium text-gray-900">{graphStats.total_nodes}</div>
              <div className="text-gray-500">Total Nodes</div>
            </div>
            <div>
              <div className="font-medium text-gray-900">{graphStats.total_edges}</div>
              <div className="text-gray-500">Total Edges</div>
            </div>
            <div>
              <div className="font-medium text-gray-900">{graphStats.avg_degree.toFixed(1)}</div>
              <div className="text-gray-500">Avg Degree</div>
            </div>
            <div>
              <div className="font-medium text-gray-900">{(graphStats.density * 100).toFixed(2)}%</div>
              <div className="text-gray-500">Density</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border-b border-red-200">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Graph Visualization */}
      <div className="p-4">
        {loading.subgraph ? (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="sm" />
            <span className="ml-2 text-sm text-gray-500">Loading citation graph...</span>
          </div>
        ) : graphData && graphData.nodes.length > 0 ? (
          <div className="space-y-4">
            {/* SVG Graph */}
            <div className="flex justify-center">
              <svg
                ref={svgRef}
                width="800"
                height="600"
                viewBox="0 0 800 600"
                className="border border-gray-200 rounded-lg bg-gray-50"
              />
            </div>
            
            {/* Graph Legend */}
            <div className="flex items-center justify-center space-x-6 text-xs text-gray-600">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-blue-600 rounded-full"></div>
                <span>Center Paper</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-green-600 rounded-full"></div>
                <span>Related Papers</span>
              </div>
              <div className="flex items-center space-x-2">
                <svg width="20" height="2" className="text-gray-400">
                  <line x1="0" y1="1" x2="20" y2="1" stroke="currentColor" markerEnd="url(#arrow)" />
                </svg>
                <span>Citations</span>
              </div>
            </div>
            
            {/* Node List */}
            <div className="mt-6">
              <h4 className="text-sm font-medium text-gray-900 mb-3">Papers in Graph</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {graphData.nodes.map((node, index) => (
                  <div
                    key={node.id}
                    className={clsx(
                      'p-2 rounded border cursor-pointer hover:bg-gray-50',
                      node.id === graphData.center_paper ? 'bg-blue-50 border-blue-200' : 'bg-white border-gray-200'
                    )}
                    onClick={() => window.open(`/papers/${node.id}`, '_blank')}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900 line-clamp-2">
                          {node.properties.title}
                        </p>
                        <p className="text-xs text-gray-600 mt-1">
                          {node.properties.authors?.slice(0, 3).join(', ')}
                          {node.properties.authors?.length > 3 && ' et al.'}
                          {node.properties.year && ` • ${node.properties.year}`}
                        </p>
                      </div>
                      <ArrowTopRightOnSquareIcon className="h-4 w-4 text-gray-400 ml-2 flex-shrink-0" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="mx-auto h-12 w-12 text-gray-400 mb-4">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <h3 className="text-sm font-medium text-gray-900 mb-2">No Citation Graph Available</h3>
            <p className="text-sm text-gray-500">
              No citation connections found for this paper in the graph database.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};