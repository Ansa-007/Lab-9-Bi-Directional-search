"""
Interactive dashboard for bi-directional search visualization.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional
import json

from ..core.search import BiDirectionalSearch
from ..core.graph import Graph, SpatialGraph, WeightedGraph
from ..utils.generators import GraphGenerator
from .plotter import SearchVisualizer


class InteractiveDashboard:
    """Interactive dashboard for exploring bi-directional search algorithms."""
    
    def __init__(self, port: int = 8050):
        self.app = dash.Dash(__name__)
        self.port = port
        self.search_instance = None
        self.current_graph = None
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Bi-Directional Search Laboratory", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Control Panel
            html.Div([
                html.H2("Control Panel"),
                
                # Graph Generation Controls
                html.Div([
                    html.H3("Graph Configuration"),
                    html.Label("Graph Type:"),
                    dcc.Dropdown(
                        id='graph-type',
                        options=[
                            {'label': 'Random Graph', 'value': 'random'},
                            {'label': 'Grid Graph', 'value': 'grid'},
                            {'label': 'Spatial Graph', 'value': 'spatial'},
                            {'label': 'Complete Graph', 'value': 'complete'}
                        ],
                        value='random'
                    ),
                    
                    html.Label("Number of Nodes:"),
                    dcc.Slider(
                        id='num-nodes',
                        min=10,
                        max=500,
                        value=50,
                        marks={i: str(i) for i in range(10, 501, 50)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Label("Edge Density:"),
                    dcc.Slider(
                        id='edge-density',
                        min=0.1,
                        max=1.0,
                        value=0.3,
                        step=0.1,
                        marks={i/10: f'{i/10:.1f}' for i in range(1, 11)},
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    ),
                    
                    html.Button('Generate Graph', id='generate-btn', 
                              n_clicks=0, style={'margin': '10px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Search Configuration
                html.Div([
                    html.H3("Search Configuration"),
                    html.Label("Search Strategy:"),
                    dcc.Dropdown(
                        id='search-strategy',
                        options=[
                            {'label': 'BFS', 'value': 'bfs'},
                            {'label': 'DFS', 'value': 'dfs'},
                            {'label': 'Dijkstra', 'value': 'dijkstra'},
                            {'label': 'A*', 'value': 'astar'}
                        ],
                        value='bfs'
                    ),
                    
                    html.Label("Start Node:"),
                    dcc.Input(id='start-node', type='number', value=0, style={'width': '100%'}),
                    
                    html.Label("Goal Node:"),
                    dcc.Input(id='goal-node', type='number', value=49, style={'width': '100%'}),
                    
                    html.Button('Run Search', id='search-btn', 
                              n_clicks=0, style={'margin': '10px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Results Panel
                html.Div([
                    html.H3("Search Results"),
                    html.Div(id='results-display', style={'padding': '10px'})
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
            
            # Visualization Tabs
            dcc.Tabs(id='tabs', value='graph-tab', children=[
                dcc.Tab(label='Graph Visualization', value='graph-tab'),
                dcc.Tab(label='Search Path', value='path-tab'),
                dcc.Tab(label='Performance Analysis', value='performance-tab'),
                dcc.Tab(label='Algorithm Comparison', value='comparison-tab')
            ]),
            
            # Tab Content
            html.Div(id='tab-content', style={'padding': '20px'}),
            
            # Store components for data sharing
            dcc.Store(id='graph-store'),
            dcc.Store(id='search-results-store'),
            dcc.Store(id='comparison-results-store')
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('graph-store', 'data'),
             Output('results-display', 'children')],
            [Input('generate-btn', 'n_clicks')],
            [State('graph-type', 'value'),
             State('num-nodes', 'value'),
             State('edge-density', 'value')]
        )
        def generate_graph(n_clicks, graph_type, num_nodes, edge_density):
            if n_clicks == 0:
                return None, "Click 'Generate Graph' to create a new graph"
            
            # Generate graph based on type
            if graph_type == 'random':
                graph = GraphGenerator.create_random_graph(num_nodes, edge_density)
            elif graph_type == 'grid':
                graph = GraphGenerator.create_grid_graph(int(np.sqrt(num_nodes)))
            elif graph_type == 'spatial':
                graph = GraphGenerator.create_spatial_graph(num_nodes, edge_density)
            elif graph_type == 'complete':
                graph = GraphGenerator.create_complete_graph(num_nodes)
            else:
                graph = GraphGenerator.create_random_graph(num_nodes, edge_density)
            
            # Store graph data
            graph_data = self._graph_to_dict(graph)
            
            results_display = html.Div([
                html.P(f"Generated {graph_type} graph"),
                html.P(f"Nodes: {len(graph.get_nodes())}"),
                html.P(f"Edges: {sum(len(list(graph.get_neighbors(n))) for n in graph.get_nodes()) // 2}")
            ])
            
            return graph_data, results_display
        
        @self.app.callback(
            [Output('search-results-store', 'data'),
             Output('results-display', 'children', allow_duplicate=True)],
            [Input('search-btn', 'n_clicks')],
            [State('graph-store', 'data'),
             State('search-strategy', 'value'),
             State('start-node', 'value'),
             State('goal-node', 'value')],
            prevent_initial_call=True
        )
        def run_search(n_clicks, graph_data, strategy, start_node, goal_node):
            if n_clicks == 0 or graph_data is None:
                return None, "No graph available for search"
            
            # Reconstruct graph from stored data
            graph = self._dict_to_graph(graph_data)
            
            # Initialize search
            self.search_instance = BiDirectionalSearch(graph)
            
            # Run search
            result = self.search_instance.search(start_node, goal_node, strategy)
            
            # Store results
            result_data = {
                'path': result.path if result.success else None,
                'path_cost': result.path_cost,
                'nodes_explored': result.nodes_explored,
                'nodes_explored_forward': result.nodes_explored_forward,
                'nodes_explored_backward': result.nodes_explored_backward,
                'time_taken': result.time_taken,
                'success': result.success,
                'meeting_point': result.meeting_point,
                'forward_depth': result.forward_depth,
                'backward_depth': result.backward_depth
            }
            
            # Display results
            if result.success:
                results_display = html.Div([
                    html.H4("Search Successful!", style={'color': 'green'}),
                    html.P(f"Path found: {' -> '.join(map(str, result.path))}"),
                    html.P(f"Path cost: {result.path_cost:.2f}"),
                    html.P(f"Nodes explored: {result.nodes_explored}"),
                    html.P(f"Time taken: {result.time_taken:.4f}s"),
                    html.P(f"Meeting point: {result.meeting_point}"),
                    html.P(f"Forward depth: {result.forward_depth}"),
                    html.P(f"Backward depth: {result.backward_depth}")
                ])
            else:
                results_display = html.Div([
                    html.H4("Search Failed!", style={'color': 'red'}),
                    html.P("No path found between the specified nodes"),
                    html.P(f"Nodes explored: {result.nodes_explored}"),
                    html.P(f"Time taken: {result.time_taken:.4f}s")
                ])
            
            return result_data, results_display
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('tabs', 'value')],
            [State('graph-store', 'data'),
             State('search-results-store', 'data')]
        )
        def update_tab_content(active_tab, graph_data, search_results):
            if active_tab == 'graph-tab':
                return self._create_graph_visualization(graph_data)
            elif active_tab == 'path-tab':
                return self._create_path_visualization(graph_data, search_results)
            elif active_tab == 'performance-tab':
                return self._create_performance_analysis(search_results)
            elif active_tab == 'comparison-tab':
                return self._create_comparison_tab()
            return html.Div("Select a tab to view content")
    
    def _graph_to_dict(self, graph: Graph) -> Dict[str, Any]:
        """Convert graph to dictionary for storage."""
        data = {
            'type': graph.__class__.__name__,
            'nodes': list(graph.get_nodes()),
            'edges': []
        }
        
        for node in graph.get_nodes():
            for neighbor, weight in graph.get_neighbors(node):
                if node < neighbor:  # Avoid duplicate edges
                    data['edges'].append({'from': node, 'to': neighbor, 'weight': weight})
        
        # Add spatial coordinates if available
        if hasattr(graph, 'coordinates'):
            data['coordinates'] = graph.coordinates
        
        return data
    
    def _dict_to_graph(self, data: Dict[str, Any]) -> Graph:
        """Reconstruct graph from dictionary."""
        if data['type'] == 'SpatialGraph':
            graph = SpatialGraph()
            # Add coordinates
            if 'coordinates' in data:
                for node, coords in data['coordinates'].items():
                    graph.add_node_with_coords(node, coords[0], coords[1])
        elif data['type'] == 'WeightedGraph':
            graph = WeightedGraph()
        else:
            graph = WeightedGraph()  # Default to weighted graph
        
        # Add edges
        for edge in data['edges']:
            graph.add_edge(edge['from'], edge['to'], edge['weight'])
        
        return graph
    
    def _create_graph_visualization(self, graph_data: Optional[Dict[str, Any]]) -> html.Div:
        """Create graph visualization tab."""
        if graph_data is None:
            return html.Div("Generate a graph first")
        
        # Create networkx graph for visualization
        G = nx.Graph()
        for node in graph_data['nodes']:
            G.add_node(node)
        
        for edge in graph_data['edges']:
            G.add_edge(edge['from'], edge['to'], weight=edge['weight'])
        
        # Calculate layout
        if 'coordinates' in graph_data:
            pos = graph_data['coordinates']
        else:
            pos = nx.spring_layout(G)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Node {node}')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="Node Connections"
                )
            )
        )
        
        # Color nodes by degree
        node_adjacencies = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
        
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Graph Structure',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(
                               text="Graph visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return html.Div([
            dcc.Graph(figure=fig)
        ])
    
    def _create_path_visualization(self, graph_data: Optional[Dict[str, Any]], 
                                  search_results: Optional[Dict[str, Any]]) -> html.Div:
        """Create search path visualization."""
        if graph_data is None:
            return html.Div("Generate a graph first")
        if search_results is None:
            return html.Div("Run a search first")
        
        if not search_results['success']:
            return html.Div("Search failed - no path to visualize")
        
        # Create visualization similar to graph tab but highlight path
        G = nx.Graph()
        for node in graph_data['nodes']:
            G.add_node(node)
        
        for edge in graph_data['edges']:
            G.add_edge(edge['from'], edge['to'], weight=edge['weight'])
        
        # Layout
        if 'coordinates' in graph_data:
            pos = graph_data['coordinates']
        else:
            pos = nx.spring_layout(G)
        
        # Separate edges into path and non-path
        path = search_results['path']
        path_edges = set(zip(path, path[1:])) | set(zip(path[1:], path))
        
        # Non-path edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            if edge not in path_edges and (edge[1], edge[0]) not in path_edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        non_path_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Path edges
        path_edge_x = []
        path_edge_y = []
        for edge in G.edges():
            if edge in path_edges or (edge[1], edge[0]) in path_edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                path_edge_x.extend([x0, x1, None])
                path_edge_y.extend([y0, y1, None])
        
        path_trace = go.Scatter(
            x=path_edge_x, y=path_edge_y,
            line=dict(width=3, color='red'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Nodes with colors
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == path[0]:  # Start node
                node_colors.append('green')
            elif node == path[-1]:  # Goal node
                node_colors.append('red')
            elif node in path:  # Path node
                node_colors.append('orange')
            else:  # Unexplored node
                node_colors.append('lightblue')
            
            node_text.append(f'Node {node}')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=15,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )
        
        fig = go.Figure(data=[non_path_trace, path_trace, node_trace],
                       layout=go.Layout(
                           title=f'Search Path (Cost: {search_results["path_cost"]:.2f})',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return html.Div([
            dcc.Graph(figure=fig)
        ])
    
    def _create_performance_analysis(self, search_results: Optional[Dict[str, Any]]) -> html.Div:
        """Create performance analysis tab."""
        if search_results is None:
            return html.Div("Run a search first")
        
        # Create performance metrics visualization
        metrics = {
            'Nodes Explored': search_results['nodes_explored'],
            'Forward Nodes': search_results['nodes_explored_forward'],
            'Backward Nodes': search_results['nodes_explored_backward'],
            'Time (ms)': search_results['time_taken'] * 1000,
            'Path Cost': search_results['path_cost']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        
        fig.update_layout(
            title='Search Performance Metrics',
            xaxis_title='Metric',
            yaxis_title='Value',
            showlegend=False
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.H4("Search Statistics"),
            html.P(f"Total nodes explored: {search_results['nodes_explored']}"),
            html.P(f"Forward search explored: {search_results['nodes_explored_forward']} nodes"),
            html.P(f"Backward search explored: {search_results['nodes_explored_backward']} nodes"),
            html.P(f"Search efficiency: {search_results['nodes_explored_forward'] + search_results['nodes_explored_backward']}/{search_results['nodes_explored']}"),
            html.P(f"Meeting point: {search_results['meeting_point']}"),
            html.P(f"Forward depth: {search_results['forward_depth']}"),
            html.P(f"Backward depth: {search_results['backward_depth']}")
        ])
    
    def _create_comparison_tab(self) -> html.Div:
        """Create algorithm comparison tab."""
        return html.Div([
            html.H4("Algorithm Comparison"),
            html.P("Run multiple searches with different strategies to compare performance."),
            html.Button('Run Comparison', id='comparison-btn', n_clicks=0),
            html.Div(id='comparison-results')
        ])
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')
