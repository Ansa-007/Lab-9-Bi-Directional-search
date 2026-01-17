"""
Professional Bi-Directional Search Laboratory

A comprehensive framework for bi-directional search algorithms with
advanced visualization, performance benchmarking, and interactive demonstrations.
"""

from .core.search import BiDirectionalSearch
from .core.graph import Graph, WeightedGraph, DirectedGraph
from .utils.generators import GraphGenerator
from .benchmark.performance import PerformanceBenchmark

__version__ = "1.0.0"
__author__ = "Professional AI Laboratory"

__all__ = [
    "BiDirectionalSearch",
    "Graph", 
    "WeightedGraph",
    "DirectedGraph",
    "GraphGenerator",
    "PerformanceBenchmark"
]
