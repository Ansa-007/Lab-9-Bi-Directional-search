"""
Graph generation utilities for testing and benchmarking.
"""

import numpy as np
import random
from typing import List, Tuple, Any, Optional
from ..core.graph import Graph, WeightedGraph, SpatialGraph, DirectedGraph


class GraphGenerator:
    """Professional graph generation utilities."""
    
    @staticmethod
    def create_random_graph(num_nodes: int, edge_probability: float, 
                           weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph:
        """Generate a random graph with specified edge probability."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = WeightedGraph()
        
        # Add all nodes
        for i in range(num_nodes):
            graph.add_node(i)
        
        # Add edges based on probability
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < edge_probability:
                    weight = np.random.uniform(1, 10) if weighted else 1.0
                    graph.add_edge(i, j, weight)
        
        return graph
    
    @staticmethod
    def create_grid_graph(rows: int, cols: Optional[int] = None, 
                          weighted: bool = True) -> WeightedGraph:
        """Generate a grid graph."""
        if cols is None:
            cols = rows
        
        graph = WeightedGraph()
        
        # Add nodes
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                graph.add_node(node_id)
        
        # Add edges (4-connected grid)
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                
                # Right neighbor
                if j < cols - 1:
                    right_id = i * cols + (j + 1)
                    weight = np.random.uniform(1, 5) if weighted else 1.0
                    graph.add_edge(node_id, right_id, weight)
                
                # Bottom neighbor
                if i < rows - 1:
                    bottom_id = (i + 1) * cols + j
                    weight = np.random.uniform(1, 5) if weighted else 1.0
                    graph.add_edge(node_id, bottom_id, weight)
        
        return graph
    
    @staticmethod
    def create_spatial_graph(num_nodes: int, edge_probability: float,
                            space_size: float = 100.0, seed: Optional[int] = None) -> SpatialGraph:
        """Generate a spatial graph with random coordinates."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = SpatialGraph()
        
        # Generate random coordinates
        coordinates = np.random.uniform(0, space_size, (num_nodes, 2))
        
        # Add nodes with coordinates
        for i in range(num_nodes):
            graph.add_node_with_coords(i, coordinates[i][0], coordinates[i][1])
        
        # Add edges based on probability
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < edge_probability:
                    graph.add_edge(i, j)  # Weight calculated automatically
        
        return graph
    
    @staticmethod
    def create_complete_graph(num_nodes: int, weighted: bool = True,
                            seed: Optional[int] = None) -> WeightedGraph:
        """Generate a complete graph."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = WeightedGraph()
        
        # Add all nodes
        for i in range(num_nodes):
            graph.add_node(i)
        
        # Add all possible edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = np.random.uniform(1, 10) if weighted else 1.0
                graph.add_edge(i, j, weight)
        
        return graph
    
    @staticmethod
    def create_tree_graph(num_nodes: int, branching_factor: int = 2,
                         weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph:
        """Generate a tree graph with specified branching factor."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = WeightedGraph()
        graph.add_node(0)  # Root node
        
        nodes_added = 1
        current_level = [0]
        
        while nodes_added < num_nodes:
            next_level = []
            for parent in current_level:
                for _ in range(branching_factor):
                    if nodes_added >= num_nodes:
                        break
                    
                    child = nodes_added
                    graph.add_node(child)
                    weight = np.random.uniform(1, 5) if weighted else 1.0
                    graph.add_edge(parent, child, weight)
                    next_level.append(child)
                    nodes_added += 1
                
                if nodes_added >= num_nodes:
                    break
            
            current_level = next_level
        
        return graph
    
    @staticmethod
    def create_scale_free_network(num_nodes: int, m: int = 2,
                                weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph:
        """Generate a scale-free network using Barabási-Albert model."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = WeightedGraph()
        
        # Start with a small complete graph
        initial_nodes = min(m + 1, num_nodes)
        for i in range(initial_nodes):
            graph.add_node(i)
        
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                weight = np.random.uniform(1, 5) if weighted else 1.0
                graph.add_edge(i, j, weight)
        
        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, num_nodes):
            graph.add_node(new_node)
            
            # Calculate degrees for preferential attachment
            degrees = []
            nodes = list(range(new_node))
            for node in nodes:
                degrees.append(graph.degree(node))
            
            total_degree = sum(degrees)
            if total_degree == 0:
                probabilities = [1.0 / len(nodes)] * len(nodes)
            else:
                probabilities = [deg / total_degree for deg in degrees]
            
            # Select m nodes to connect to
            selected_nodes = np.random.choice(nodes, size=min(m, len(nodes)), 
                                            replace=False, p=probabilities)
            
            for selected in selected_nodes:
                weight = np.random.uniform(1, 5) if weighted else 1.0
                graph.add_edge(new_node, selected, weight)
        
        return graph
    
    @staticmethod
    def create_small_world_network(num_nodes: int, k: int = 4, p: float = 0.3,
                                  weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph:
        """Generate a small-world network using Watts-Strogatz model."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = WeightedGraph()
        
        # Add all nodes
        for i in range(num_nodes):
            graph.add_node(i)
        
        # Create regular ring lattice
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % num_nodes
                weight = np.random.uniform(1, 5) if weighted else 1.0
                graph.add_edge(i, neighbor, weight)
        
        # Rewire edges with probability p
        edges_to_rewire = []
        for i in range(num_nodes):
            for j in range(1, k // 2 + 1):
                if np.random.random() < p:
                    neighbor = (i + j) % num_nodes
                    edges_to_rewire.append((i, neighbor))
        
        # Rewire edges
        for u, v in edges_to_rewire:
            # Remove old edge
            if v in graph.adjacency_list[u]:
                graph.adjacency_list[u].remove(v)
            if u in graph.adjacency_list[v]:
                graph.adjacency_list[v].remove(u)
            
            # Add new edge to random node
            possible_targets = [n for n in range(num_nodes) if n != u and n not in graph.adjacency_list[u]]
            if possible_targets:
                new_target = random.choice(possible_targets)
                weight = np.random.uniform(1, 5) if weighted else 1.0
                graph.add_edge(u, new_target, weight)
        
        return graph
    
    @staticmethod
    def create_maze_graph(rows: int, cols: Optional[int] = None) -> WeightedGraph:
        """Generate a maze-like graph (perfect maze)."""
        if cols is None:
            cols = rows
        
        graph = WeightedGraph()
        
        # Add all nodes
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                graph.add_node(node_id)
        
        # Use recursive backtracking to generate maze
        visited = set()
        stack = []
        start_node = 0
        visited.add(start_node)
        stack.append(start_node)
        
        while stack:
            current = stack[-1]
            row, col = current // cols, current % cols
            
            # Find unvisited neighbors
            neighbors = []
            if row > 0:  # Top
                top = (row - 1) * cols + col
                if top not in visited:
                    neighbors.append(top)
            if row < rows - 1:  # Bottom
                bottom = (row + 1) * cols + col
                if bottom not in visited:
                    neighbors.append(bottom)
            if col > 0:  # Left
                left = row * cols + (col - 1)
                if left not in visited:
                    neighbors.append(left)
            if col < cols - 1:  # Right
                right = row * cols + (col + 1)
                if right not in visited:
                    neighbors.append(right)
            
            if neighbors:
                next_node = random.choice(neighbors)
                visited.add(next_node)
                stack.append(next_node)
                graph.add_edge(current, next_node, 1.0)
            else:
                stack.pop()
        
        return graph
    
    @staticmethod
    def create_directed_graph(num_nodes: int, edge_probability: float,
                             weighted: bool = True, seed: Optional[int] = None) -> DirectedGraph:
        """Generate a random directed graph."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = DirectedGraph(weighted=weighted)
        
        # Add all nodes
        for i in range(num_nodes):
            graph.add_node(i)
        
        # Add directed edges based on probability
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and np.random.random() < edge_probability:
                    weight = np.random.uniform(1, 10) if weighted else None
                    graph.add_edge(i, j, weight)
        
        return graph
    
    @staticmethod
    def create_bipartite_graph(size_left: int, size_right: int,
                              edge_probability: float, weighted: bool = True,
                              seed: Optional[int] = None) -> WeightedGraph:
        """Generate a bipartite graph."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        graph = WeightedGraph()
        
        # Add left partition nodes
        for i in range(size_left):
            graph.add_node(f"L{i}")
        
        # Add right partition nodes
        for i in range(size_right):
            graph.add_node(f"R{i}")
        
        # Add edges between partitions
        for i in range(size_left):
            for j in range(size_right):
                if np.random.random() < edge_probability:
                    weight = np.random.uniform(1, 10) if weighted else 1.0
                    graph.add_edge(f"L{i}", f"R{j}", weight)
        
        return graph
    
    @staticmethod
    def create_custom_graph(adjacency_matrix: np.ndarray,
                           node_labels: Optional[List[Any]] = None) -> WeightedGraph:
        """Create graph from adjacency matrix."""
        n = adjacency_matrix.shape[0]
        if adjacency_matrix.shape[1] != n:
            raise ValueError("Adjacency matrix must be square")
        
        graph = WeightedGraph()
        
        # Add nodes
        if node_labels is None:
            node_labels = list(range(n))
        
        for label in node_labels:
            graph.add_node(label)
        
        # Add edges
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] > 0:
                    graph.add_edge(node_labels[i], node_labels[j], adjacency_matrix[i, j])
        
        return graph
