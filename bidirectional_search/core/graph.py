"""
Graph data structures for bi-directional search algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from collections import defaultdict


class Graph(ABC):
    """Abstract base class for graph data structures."""
    
    def __init__(self):
        self.adjacency_list: Dict[Any, List[Any]] = defaultdict(list)
        self.nodes: Set[Any] = set()
    
    @abstractmethod
    def add_edge(self, u: Any, v: Any, weight: Optional[float] = None) -> None:
        """Add an edge between nodes u and v."""
        pass
    
    @abstractmethod
    def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """Get neighbors of a node with their edge weights."""
        pass
    
    def add_node(self, node: Any) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)
    
    def has_node(self, node: Any) -> bool:
        """Check if node exists in graph."""
        return node in self.nodes
    
    def get_nodes(self) -> Set[Any]:
        """Get all nodes in the graph."""
        return self.nodes.copy()
    
    def degree(self, node: Any) -> int:
        """Get degree of a node."""
        return len(self.adjacency_list[node])
    
    def size(self) -> int:
        """Get number of nodes in the graph."""
        return len(self.nodes)


class UnweightedGraph(Graph):
    """Unweighted graph implementation."""
    
    def add_edge(self, u: Any, v: Any, weight: Optional[float] = None) -> None:
        """Add an unweighted edge between nodes u and v."""
        self.add_node(u)
        self.add_node(v)
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)  # Undirected
    
    def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """Get neighbors with unit weight."""
        return [(neighbor, 1.0) for neighbor in self.adjacency_list[node]]


class WeightedGraph(Graph):
    """Weighted graph implementation."""
    
    def __init__(self):
        super().__init__()
        self.weights: Dict[Tuple[Any, Any], float] = {}
    
    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """Add a weighted edge between nodes u and v."""
        self.add_node(u)
        self.add_node(v)
        self.adjacency_list[u].append(v)
        self.adjacency_list[v].append(u)  # Undirected
        self.weights[(u, v)] = weight
        self.weights[(v, u)] = weight
    
    def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """Get neighbors with their edge weights."""
        return [(neighbor, self.weights[(node, neighbor)]) 
                for neighbor in self.adjacency_list[node]]
    
    def get_weight(self, u: Any, v: Any) -> float:
        """Get weight of edge between u and v."""
        return self.weights.get((u, v), float('inf'))


class DirectedGraph(Graph):
    """Directed graph implementation."""
    
    def __init__(self, weighted: bool = False):
        super().__init__()
        self.weights: Dict[Tuple[Any, Any], float] = {}
        self.weighted = weighted
    
    def add_edge(self, u: Any, v: Any, weight: Optional[float] = None) -> None:
        """Add a directed edge from u to v."""
        self.add_node(u)
        self.add_node(v)
        self.adjacency_list[u].append(v)
        
        if self.weighted and weight is not None:
            self.weights[(u, v)] = weight
        elif self.weighted:
            self.weights[(u, v)] = 1.0
    
    def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """Get outgoing neighbors with their edge weights."""
        if self.weighted:
            return [(neighbor, self.weights.get((node, neighbor), 1.0)) 
                    for neighbor in self.adjacency_list[node]]
        return [(neighbor, 1.0) for neighbor in self.adjacency_list[node]]
    
    def get_incoming_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """Get incoming neighbors for directed graphs."""
        incoming = []
        for other_node in self.nodes:
            if node in self.adjacency_list[other_node]:
                if self.weighted:
                    weight = self.weights.get((other_node, node), 1.0)
                    incoming.append((other_node, weight))
                else:
                    incoming.append((other_node, 1.0))
        return incoming


class SpatialGraph(WeightedGraph):
    """Graph with spatial coordinates for geometric heuristics."""
    
    def __init__(self):
        super().__init__()
        self.coordinates: Dict[Any, Tuple[float, float]] = {}
    
    def add_node_with_coords(self, node: Any, x: float, y: float) -> None:
        """Add a node with spatial coordinates."""
        self.add_node(node)
        self.coordinates[node] = (x, y)
    
    def get_coordinates(self, node: Any) -> Optional[Tuple[float, float]]:
        """Get coordinates of a node."""
        return self.coordinates.get(node)
    
    def add_edge(self, u: Any, v: Any, weight: Optional[float] = None) -> None:
        """Add edge with automatic weight calculation based on distance."""
        if weight is None and u in self.coordinates and v in self.coordinates:
            # Calculate Euclidean distance as default weight
            x1, y1 = self.coordinates[u]
            x2, y2 = self.coordinates[v]
            weight = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        elif weight is None:
            weight = 1.0
        
        super().add_edge(u, v, weight)
