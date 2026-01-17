"""
Static visualization components for bi-directional search.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns

from ..core.search import BiDirectionalSearch, SearchResult
from ..core.graph import Graph, SpatialGraph


class SearchVisualizer:
    """Visualizer for bi-directional search process and results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_search_comparison(self, results: Dict[str, SearchResult], 
                              title: str = "Search Algorithm Comparison") -> plt.Figure:
        """Compare performance of different search algorithms."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        algorithms = list(results.keys())
        
        # Nodes explored comparison
        nodes_explored = [results[alg].nodes_explored for alg in algorithms]
        axes[0, 0].bar(algorithms, nodes_explored, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Nodes Explored')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Time taken comparison
        time_taken = [results[alg].time_taken for alg in algorithms]
        axes[0, 1].bar(algorithms, time_taken, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Time Taken')
        axes[0, 1].set_ylabel('Seconds')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Path cost comparison
        path_costs = [results[alg].path_cost if results[alg].success else 0 
                     for alg in algorithms]
        axes[1, 0].bar(algorithms, path_costs, color='salmon', alpha=0.7)
        axes[1, 0].set_title('Path Cost')
        axes[1, 0].set_ylabel('Cost')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Success rate
        success_rates = [1 if results[alg].success else 0 for alg in algorithms]
        axes[1, 1].bar(algorithms, success_rates, color='gold', alpha=0.7)
        axes[1, 1].set_title('Success Rate')
        axes[1, 1].set_ylabel('Success (1/0)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_search_progression(self, search_instance: BiDirectionalSearch,
                               start: Any, goal: Any, strategy: str = "bfs") -> plt.Figure:
        """Visualize the step-by-step progression of bi-directional search."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Perform search and capture intermediate states
        result = search_instance.search(start, goal, strategy)
        
        # Left plot: Search frontier sizes over time
        ax1.set_title(f'Search Frontier Sizes - {strategy.upper()}')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Frontier Size')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Meeting point visualization
        ax2.set_title(f'Search Meeting Point - {strategy.upper()}')
        ax2.set_xlabel('Forward Depth')
        ax2.set_ylabel('Backward Depth')
        ax2.grid(True, alpha=0.3)
        
        if result.success and result.meeting_point:
            ax2.scatter(result.forward_depth, result.backward_depth, 
                       s=200, c='red', marker='*', label='Meeting Point')
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_efficiency_analysis(self, graph_sizes: List[int], 
                                 results: Dict[int, Dict[str, SearchResult]]) -> plt.Figure:
        """Analyze search efficiency across different graph sizes."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Bi-Directional Search Efficiency Analysis', 
                    fontsize=16, fontweight='bold')
        
        strategies = list(next(iter(results.values())).keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        # Time complexity
        ax = axes[0, 0]
        for i, strategy in enumerate(strategies):
            times = [results[size][strategy].time_taken for size in graph_sizes]
            ax.plot(graph_sizes, times, 'o-', label=strategy, color=colors[i], linewidth=2)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Space complexity
        ax = axes[0, 1]
        for i, strategy in enumerate(strategies):
            nodes = [results[size][strategy].nodes_explored for size in graph_sizes]
            ax.plot(graph_sizes, nodes, 's-', label=strategy, color=colors[i], linewidth=2)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Nodes Explored')
        ax.set_title('Space Complexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Success rate
        ax = axes[1, 0]
        for i, strategy in enumerate(strategies):
            success_rates = [1 if results[size][strategy].success else 0 
                           for size in graph_sizes]
            ax.plot(graph_sizes, success_rates, '^-', label=strategy, 
                   color=colors[i], linewidth=2)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate vs Graph Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Path quality
        ax = axes[1, 1]
        for i, strategy in enumerate(strategies):
            path_costs = [results[size][strategy].path_cost if results[size][strategy].success else np.nan 
                         for size in graph_sizes]
            ax.plot(graph_sizes, path_costs, 'd-', label=strategy, 
                   color=colors[i], linewidth=2)
        ax.set_xlabel('Graph Size (nodes)')
        ax.set_ylabel('Path Cost')
        ax.set_title('Path Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        return fig


class GraphPlotter:
    """Graph visualization utilities."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
    
    def plot_graph(self, graph: Graph, layout: str = 'spring', 
                   node_colors: Optional[Dict[Any, str]] = None,
                   edge_colors: Optional[Dict[Tuple[Any, Any], str]] = None) -> plt.Figure:
        """Plot graph structure."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to networkx graph for visualization
        G = nx.Graph()
        
        # Add nodes
        for node in graph.get_nodes():
            G.add_node(node)
        
        # Add edges
        for node in graph.get_nodes():
            for neighbor, weight in graph.get_neighbors(node):
                if node < neighbor:  # Avoid duplicate edges in undirected graphs
                    G.add_edge(node, neighbor, weight=weight)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Default node colors
        if node_colors is None:
            node_colors = {node: 'lightblue' for node in G.nodes()}
        
        # Draw graph
        node_color_list = [node_colors.get(node, 'lightblue') for node in G.nodes()]
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color_list,
               node_size=500, font_size=8, font_weight='bold',
               edge_color='gray', width=1, alpha=0.7)
        
        ax.set_title('Graph Structure', fontsize=14, fontweight='bold')
        return fig
    
    def plot_search_path(self, graph: Graph, result: SearchResult,
                        start: Any, goal: Any, layout: str = 'spring') -> plt.Figure:
        """Visualize the search path on the graph."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to networkx graph
        G = nx.Graph()
        for node in graph.get_nodes():
            G.add_node(node)
        
        for node in graph.get_nodes():
            for neighbor, weight in graph.get_neighbors(node):
                if node < neighbor:
                    G.add_edge(node, neighbor, weight=weight)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Node colors
        node_colors = {}
        for node in G.nodes():
            if node == start:
                node_colors[node] = 'green'
            elif node == goal:
                node_colors[node] = 'red'
            elif result.success and node in result.path:
                node_colors[node] = 'orange'
            else:
                node_colors[node] = 'lightblue'
        
        # Edge colors
        edge_colors = []
        edge_widths = []
        for edge in G.edges():
            if result.success and edge in zip(result.path, result.path[1:]):
                edge_colors.append('red')
                edge_widths.append(3)
            elif result.success and (edge[1], edge[0]) in zip(result.path, result_path[1:]):
                edge_colors.append('red')
                edge_widths.append(3)
            else:
                edge_colors.append('gray')
                edge_widths.append(1)
        
        # Draw graph
        node_color_list = [node_colors.get(node, 'lightblue') for node in G.nodes()]
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color_list,
               node_size=500, font_size=8, font_weight='bold',
               edge_color=edge_colors, width=edge_widths, alpha=0.7)
        
        # Add legend
        legend_elements = [
            patches.Patch(color='green', label='Start'),
            patches.Patch(color='red', label='Goal'),
            patches.Patch(color='orange', label='Path'),
            patches.Patch(color='lightblue', label='Unexplored')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        title = f'Search Path - {result.nodes_explored} nodes explored'
        if result.success:
            title += f' (Cost: {result.path_cost:.2f})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig
    
    def plot_spatial_graph(self, graph: SpatialGraph, result: Optional[SearchResult] = None,
                          start: Optional[Any] = None, goal: Optional[Any] = None) -> plt.Figure:
        """Plot spatial graph with coordinates."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract coordinates
        coords = {}
        for node in graph.get_nodes():
            coord = graph.get_coordinates(node)
            if coord:
                coords[node] = coord
        
        if not coords:
            ax.text(0.5, 0.5, 'No spatial coordinates available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Plot edges
        for node in graph.get_nodes():
            node_coord = coords.get(node)
            if node_coord:
                for neighbor, weight in graph.get_neighbors(node):
                    neighbor_coord = coords.get(neighbor)
                    if neighbor_coord:
                        x_vals = [node_coord[0], neighbor_coord[0]]
                        y_vals = [node_coord[1], neighbor_coord[1]]
                        
                        # Determine edge color and width
                        if result and result.success and node in result.path and neighbor in result.path:
                            edge_color = 'red'
                            edge_width = 2
                        else:
                            edge_color = 'gray'
                            edge_width = 0.5
                        
                        ax.plot(x_vals, y_vals, color=edge_color, linewidth=edge_width, alpha=0.6)
        
        # Plot nodes
        x_coords = [coord[0] for coord in coords.values()]
        y_coords = [coord[1] for coord in coords.values()]
        
        # Determine node colors
        node_colors = []
        for node in coords.keys():
            if node == start:
                node_colors.append('green')
            elif node == goal:
                node_colors.append('red')
            elif result and result.success and node in result.path:
                node_colors.append('orange')
            else:
                node_colors.append('lightblue')
        
        ax.scatter(x_coords, y_coords, c=node_colors, s=100, alpha=0.8, edgecolors='black')
        
        # Add node labels
        for node, (x, y) in coords.items():
            ax.annotate(str(node), (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Spatial Graph Visualization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend if needed
        if start or goal or result:
            legend_elements = []
            if start:
                legend_elements.append(patches.Patch(color='green', label='Start'))
            if goal:
                legend_elements.append(patches.Patch(color='red', label='Goal'))
            if result and result.success:
                legend_elements.append(patches.Patch(color='orange', label='Path'))
            legend_elements.append(patches.Patch(color='lightblue', label='Unexplored'))
            ax.legend(handles=legend_elements, loc='upper right')
        
        return fig
