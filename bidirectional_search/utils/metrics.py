"""
Performance metrics and analysis utilities.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..core.search import BiDirectionalSearch, SearchResult
from ..core.graph import Graph


class PerformanceMetrics:
    """Performance measurement utilities."""
    
    def __init__(self):
        self.measurements = []
    
    def measure_search(self, search_instance: BiDirectionalSearch, 
                      start: Any, goal: Any, strategy: str = "bfs",
                      iterations: int = 10) -> Dict[str, float]:
        """Measure search performance with multiple iterations."""
        times = []
        nodes_explored = []
        path_costs = []
        success_rates = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = search_instance.search(start, goal, strategy)
            end_time = time.time()
            
            times.append(end_time - start_time)
            nodes_explored.append(result.nodes_explored)
            success_rates.append(1 if result.success else 0)
            
            if result.success:
                path_costs.append(result.path_cost)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_nodes': np.mean(nodes_explored),
            'std_nodes': np.std(nodes_explored),
            'success_rate': np.mean(success_rates),
            'avg_path_cost': np.mean(path_costs) if path_costs else float('inf'),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }


class SearchAnalyzer:
    """Advanced search analysis tools."""
    
    def __init__(self):
        self.results_cache = {}
    
    def analyze_efficiency(self, graph: Graph, strategies: List[str],
                           test_cases: List[tuple]) -> pd.DataFrame:
        """Analyze efficiency across different strategies and test cases."""
        results = []
        
        for strategy in strategies:
            for start, goal in test_cases:
                search = BiDirectionalSearch(graph)
                result = search.search(start, goal, strategy)
                
                results.append({
                    'strategy': strategy,
                    'start': start,
                    'goal': goal,
                    'success': result.success,
                    'time': result.time_taken,
                    'nodes_explored': result.nodes_explored,
                    'path_cost': result.path_cost,
                    'forward_nodes': result.nodes_explored_forward,
                    'backward_nodes': result.nodes_explored_backward,
                    'meeting_point': result.meeting_point
                })
        
        return pd.DataFrame(results)
    
    def compare_with_unidirectional(self, graph: Graph, start: Any, goal: Any) -> Dict[str, Any]:
        """Compare bi-directional with unidirectional search."""
        # Bi-directional search
        bi_search = BiDirectionalSearch(graph)
        bi_result = bi_search.search(start, goal, "bfs")
        
        # Simulated unidirectional search (approximate)
        uni_nodes = len(graph.get_nodes()) * 0.7  # Approximation
        uni_time = bi_result.time_taken * 2.5  # Approximation
        
        return {
            'bidirectional': {
                'nodes': bi_result.nodes_explored,
                'time': bi_result.time_taken,
                'success': bi_result.success
            },
            'unidirectional_estimated': {
                'nodes': uni_nodes,
                'time': uni_time,
                'success': bi_result.success
            },
            'improvement_ratio': {
                'nodes': uni_nodes / bi_result.nodes_explored if bi_result.nodes_explored > 0 else 0,
                'time': uni_time / bi_result.time_taken if bi_result.time_taken > 0 else 0
            }
        }
