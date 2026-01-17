"""
Visualization components for bi-directional search algorithms.
"""

from .plotter import SearchVisualizer, GraphPlotter

try:
    from .dashboard import InteractiveDashboard
    __all__ = [
        "SearchVisualizer",
        "GraphPlotter", 
        "InteractiveDashboard"
    ]
except ImportError:
    __all__ = [
        "SearchVisualizer",
        "GraphPlotter"
    ]
