"""
Heuristic functions for informed bi-directional search.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np
from .graph import SpatialGraph


class HeuristicFunction(ABC):
    """Abstract base class for heuristic functions."""
    
    @abstractmethod
    def __call__(self, node: Any, goal: Any) -> float:
        """Calculate heuristic value from node to goal."""
        pass


class EuclideanDistance(HeuristicFunction):
    """Euclidean distance heuristic for spatial graphs."""
    
    def __init__(self, graph: SpatialGraph):
        self.graph = graph
    
    def __call__(self, node: Any, goal: Any) -> float:
        """Calculate Euclidean distance between node and goal."""
        node_coords = self.graph.get_coordinates(node)
        goal_coords = self.graph.get_coordinates(goal)
        
        if node_coords is None or goal_coords is None:
            return float('inf')
        
        x1, y1 = node_coords
        x2, y2 = goal_coords
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class ManhattanDistance(HeuristicFunction):
    """Manhattan distance heuristic for spatial graphs."""
    
    def __init__(self, graph: SpatialGraph):
        self.graph = graph
    
    def __call__(self, node: Any, goal: Any) -> float:
        """Calculate Manhattan distance between node and goal."""
        node_coords = self.graph.get_coordinates(node)
        goal_coords = self.graph.get_coordinates(goal)
        
        if node_coords is None or goal_coords is None:
            return float('inf')
        
        x1, y1 = node_coords
        x2, y2 = goal_coords
        return abs(x2 - x1) + abs(y2 - y1)


class ZeroHeuristic(HeuristicFunction):
    """Zero heuristic (uninformed search)."""
    
    def __call__(self, node: Any, goal: Any) -> float:
        """Always return 0."""
        return 0.0


class DegreeHeuristic(HeuristicFunction):
    """Heuristic based on node degree (inverse)."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def __call__(self, node: Any, goal: Any) -> float:
        """Return inverse of node degree."""
        if not self.graph.has_node(node):
            return float('inf')
        degree = self.graph.degree(node)
        return 1.0 / (degree + 1)  # +1 to avoid division by zero


class RandomHeuristic(HeuristicFunction):
    """Random heuristic for testing purposes."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, node: Any, goal: Any) -> float:
        """Return random heuristic value."""
        return self.rng.random()


class AdaptiveHeuristic(HeuristicFunction):
    """Adaptive heuristic that combines multiple heuristics."""
    
    def __init__(self, heuristics: list, weights: Optional[list] = None):
        if weights is None:
            weights = [1.0] * len(heuristics)
        elif len(weights) != len(heuristics):
            raise ValueError("Number of weights must match number of heuristics")
        
        self.heuristics = heuristics
        self.weights = weights
    
    def __call__(self, node: Any, goal: Any) -> float:
        """Combine multiple heuristics with weights."""
        total = 0.0
        for heuristic, weight in zip(self.heuristics, self.weights):
            total += weight * heuristic(node, goal)
        return total


# Convenience functions for common heuristics
def euclidean_distance(graph: SpatialGraph) -> EuclideanDistance:
    """Create Euclidean distance heuristic."""
    return EuclideanDistance(graph)


def manhattan_distance(graph: SpatialGraph) -> ManhattanDistance:
    """Create Manhattan distance heuristic."""
    return ManhattanDistance(graph)


def zero_heuristic() -> ZeroHeuristic:
    """Create zero heuristic."""
    return ZeroHeuristic()


def degree_heuristic(graph) -> DegreeHeuristic:
    """Create degree-based heuristic."""
    return DegreeHeuristic(graph)
