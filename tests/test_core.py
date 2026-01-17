"""
Unit tests for core bi-directional search functionality.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidirectional_search.core.search import BiDirectionalSearch, SearchResult
from bidirectional_search.core.graph import Graph, WeightedGraph, SpatialGraph
from bidirectional_search.core.heuristics import EuclideanDistance, ManhattanDistance, ZeroHeuristic
from bidirectional_search.utils.generators import GraphGenerator


class TestGraph(unittest.TestCase):
    """Test graph data structures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = WeightedGraph()
        self.graph.add_edge(0, 1, 1.0)
        self.graph.add_edge(1, 2, 2.0)
        self.graph.add_edge(2, 3, 1.5)
        self.graph.add_edge(0, 3, 3.0)
    
    def test_graph_creation(self):
        """Test graph creation and basic properties."""
        self.assertEqual(len(self.graph.get_nodes()), 4)
        self.assertIn(0, self.graph.get_nodes())
        self.assertIn(3, self.graph.get_nodes())
        self.assertTrue(self.graph.has_node(1))
        self.assertFalse(self.graph.has_node(5))
    
    def test_edge_operations(self):
        """Test edge addition and neighbor retrieval."""
        neighbors = self.graph.get_neighbors(1)
        neighbor_nodes = [n[0] for n in neighbors]
        self.assertIn(0, neighbor_nodes)
        self.assertIn(2, neighbor_nodes)
        
        # Test edge weights
        edge_weights = {n[0]: n[1] for n in neighbors}
        self.assertEqual(edge_weights[0], 1.0)
        self.assertEqual(edge_weights[2], 2.0)
    
    def test_spatial_graph(self):
        """Test spatial graph functionality."""
        spatial_graph = SpatialGraph()
        spatial_graph.add_node_with_coords(0, 0.0, 0.0)
        spatial_graph.add_node_with_coords(1, 1.0, 1.0)
        spatial_graph.add_edge(0, 1)
        
        coords = spatial_graph.get_coordinates(0)
        self.assertEqual(coords, (0.0, 0.0))
        
        weight = spatial_graph.get_weight(0, 1)
        self.assertAlmostEqual(weight, 1.4142135623730951)  # sqrt(2)


class TestHeuristics(unittest.TestCase):
    """Test heuristic functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.spatial_graph = SpatialGraph()
        self.spatial_graph.add_node_with_coords(0, 0.0, 0.0)
        self.spatial_graph.add_node_with_coords(1, 3.0, 4.0)
        self.spatial_graph.add_node_with_coords(2, 1.0, 1.0)
    
    def test_euclidean_distance(self):
        """Test Euclidean distance heuristic."""
        heuristic = EuclideanDistance(self.spatial_graph)
        distance = heuristic(0, 1)
        self.assertAlmostEqual(distance, 5.0)  # 3-4-5 triangle
    
    def test_manhattan_distance(self):
        """Test Manhattan distance heuristic."""
        heuristic = ManhattanDistance(self.spatial_graph)
        distance = heuristic(0, 1)
        self.assertEqual(distance, 7.0)  # |3-0| + |4-0|
    
    def test_zero_heuristic(self):
        """Test zero heuristic."""
        heuristic = ZeroHeuristic()
        distance = heuristic(0, 1)
        self.assertEqual(distance, 0.0)


class TestBiDirectionalSearch(unittest.TestCase):
    """Test bi-directional search algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = GraphGenerator.create_grid_graph(3, 3)
        self.search = BiDirectionalSearch(self.graph)
    
    def test_basic_search(self):
        """Test basic bi-directional search."""
        result = self.search.search(0, 8, strategy="bfs")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.path)
        self.assertEqual(result.path[0], 0)
        self.assertEqual(result.path[-1], 8)
        self.assertGreater(result.nodes_explored, 0)
        self.assertGreaterEqual(result.time_taken, 0)
    
    def test_strategies(self):
        """Test different search strategies."""
        strategies = ["bfs", "dfs", "dijkstra"]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                result = self.search.search(0, 8, strategy=strategy)
                self.assertTrue(result.success, f"Strategy {strategy} failed")
                self.assertIsNotNone(result.path)
    
    def test_no_path_case(self):
        """Test case where no path exists."""
        # Create disconnected graph
        disconnected_graph = WeightedGraph()
        disconnected_graph.add_edge(0, 1, 1.0)
        disconnected_graph.add_edge(1, 2, 1.0)
        disconnected_graph.add_edge(3, 4, 1.0)
        disconnected_graph.add_edge(4, 5, 1.0)
        
        search = BiDirectionalSearch(disconnected_graph)
        result = search.search(0, 5, strategy="bfs")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.path)
        self.assertEqual(result.path_cost, float('inf'))
    
    def test_same_start_goal(self):
        """Test case where start equals goal."""
        result = self.search.search(5, 5, strategy="bfs")
        
        self.assertTrue(result.success)
        self.assertEqual(result.path, [5])
        self.assertEqual(result.path_cost, 0.0)
    
    def test_invalid_nodes(self):
        """Test case with invalid start or goal nodes."""
        result = self.search.search(0, 100, strategy="bfs")
        
        self.assertFalse(result.success)
        self.assertIsNone(result.path)
    
    def test_meeting_point(self):
        """Test meeting point detection."""
        result = self.search.search(0, 8, strategy="bfs")
        
        if result.success:
            self.assertIsNotNone(result.meeting_point)
            self.assertIn(result.meeting_point, result.path)
            self.assertGreaterEqual(result.forward_depth, 0)
            self.assertGreaterEqual(result.backward_depth, 0)


class TestSearchResult(unittest.TestCase):
    """Test SearchResult data structure."""
    
    def test_successful_result(self):
        """Test successful search result."""
        result = SearchResult(
            path=[0, 1, 2],
            path_cost=2.0,
            nodes_explored=10,
            nodes_explored_forward=5,
            nodes_explored_backward=5,
            time_taken=0.001,
            success=True,
            meeting_point=1,
            forward_depth=1,
            backward_depth=1
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.path, [0, 1, 2])
        self.assertEqual(result.path_cost, 2.0)
        self.assertEqual(result.nodes_explored, 10)
        self.assertEqual(result.meeting_point, 1)
    
    def test_failed_result(self):
        """Test failed search result."""
        result = SearchResult(
            path=None,
            path_cost=float('inf'),
            nodes_explored=20,
            nodes_explored_forward=10,
            nodes_explored_backward=10,
            time_taken=0.002,
            success=False
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.path)
        self.assertEqual(result.path_cost, float('inf'))


class TestGraphGenerators(unittest.TestCase):
    """Test graph generation utilities."""
    
    def test_random_graph(self):
        """Test random graph generation."""
        graph = GraphGenerator.create_random_graph(10, 0.3, weighted=True, seed=42)
        
        self.assertEqual(len(graph.get_nodes()), 10)
        self.assertGreater(len(list(graph.get_neighbors(0))), 0)
    
    def test_grid_graph(self):
        """Test grid graph generation."""
        graph = GraphGenerator.create_grid_graph(3, 4)
        
        self.assertEqual(len(graph.get_nodes()), 12)
        
        # Check that corner nodes have 2 neighbors
        corner_neighbors = len(list(graph.get_neighbors(0)))
        self.assertEqual(corner_neighbors, 2)
    
    def test_spatial_graph(self):
        """Test spatial graph generation."""
        graph = GraphGenerator.create_spatial_graph(10, 0.5, space_size=100, seed=42)
        
        self.assertEqual(len(graph.get_nodes()), 10)
        self.assertTrue(hasattr(graph, 'coordinates'))
        
        # Check that coordinates exist for nodes
        coords = graph.get_coordinates(0)
        self.assertIsNotNone(coords)
        self.assertEqual(len(coords), 2)
    
    def test_complete_graph(self):
        """Test complete graph generation."""
        graph = GraphGenerator.create_complete_graph(5)
        
        self.assertEqual(len(graph.get_nodes()), 5)
        
        # Each node should have 4 neighbors in a complete graph of 5 nodes
        for node in graph.get_nodes():
            neighbors = len(list(graph.get_neighbors(node)))
            self.assertEqual(neighbors, 4)
    
    def test_tree_graph(self):
        """Test tree graph generation."""
        graph = GraphGenerator.create_tree_graph(7, branching_factor=2)
        
        self.assertEqual(len(graph.get_nodes()), 7)
        
        # Tree should have exactly n-1 edges
        total_edges = sum(len(list(graph.get_neighbors(n))) for n in graph.get_nodes()) // 2
        self.assertEqual(total_edges, 6)


if __name__ == '__main__':
    unittest.main()
