#!/usr/bin/env python3
"""
Basic usage examples for the Bi-Directional Search Laboratory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidirectional_search import BiDirectionalSearch, GraphGenerator
from bidirectional_search.visualization.plotter import SearchVisualizer, GraphPlotter
import matplotlib.pyplot as plt


def example_basic_search():
    """Basic bi-directional search example."""
    print("=== Basic Bi-Directional Search Example ===")
    
    # Create a simple graph
    graph = GraphGenerator.create_random_graph(20, 0.3, weighted=True)
    print(f"Created graph with {len(graph.get_nodes())} nodes")
    
    # Initialize search
    search = BiDirectionalSearch(graph)
    
    # Find path
    start, goal = 0, 19
    result = search.find_path(start, goal, strategy="bfs")
    
    if result.success:
        print(f"Path found: {result.path}")
        print(f"Path cost: {result.path_cost:.2f}")
        print(f"Nodes explored: {result.nodes_explored}")
        print(f"Time taken: {result.time_taken:.4f}s")
        print(f"Meeting point: {result.meeting_point}")
    else:
        print("No path found")
    
    return result


def example_strategy_comparison():
    """Compare different search strategies."""
    print("\n=== Strategy Comparison Example ===")
    
    # Create graph
    graph = GraphGenerator.create_grid_graph(5, 5)
    print(f"Created 5x5 grid graph with {len(graph.get_nodes())} nodes")
    
    strategies = ["bfs", "dfs", "dijkstra", "astar"]
    results = {}
    
    for strategy in strategies:
        search = BiDirectionalSearch(graph)
        result = search.find_path(0, 24, strategy=strategy)
        results[strategy] = result
        
        print(f"\n{strategy.upper()}:")
        if result.success:
            print(f"  Path found: {' -> '.join(map(str, result.path))}")
            print(f"  Cost: {result.path_cost:.2f}")
            print(f"  Nodes explored: {result.nodes_explored}")
            print(f"  Time: {result.time_taken:.4f}s")
        else:
            print("  No path found")
    
    return results


def example_visualization():
    """Create visualizations of search results."""
    print("\n=== Visualization Example ===")
    
    # Create spatial graph for better visualization
    graph = GraphGenerator.create_spatial_graph(15, 0.4, space_size=50)
    search = BiDirectionalSearch(graph)
    
    # Run search
    result = search.find_path(0, 14, strategy="bfs")
    
    if result.success:
        # Create visualizer
        plotter = GraphPlotter()
        
        # Plot graph with path
        fig = plotter.plot_spatial_graph(graph, result, 0, 14)
        plt.title(f"Bi-Directional Search Path (Cost: {result.path_cost:.2f})")
        plt.show()
        
        print("Visualization displayed")
    else:
        print("No path found for visualization")


def example_performance_analysis():
    """Performance analysis example."""
    print("\n=== Performance Analysis Example ===")
    
    # Create visualizer
    visualizer = SearchVisualizer()
    
    # Test different graph sizes
    graph_sizes = [10, 20, 30, 40, 50]
    strategies = ["bfs", "dijkstra"]
    results = {}
    
    for size in graph_sizes:
        graph = GraphGenerator.create_random_graph(size, 0.3)
        size_results = {}
        
        for strategy in strategies:
            search = BiDirectionalSearch(graph)
            result = search.find_path(0, size-1, strategy=strategy)
            size_results[strategy] = result
        
        results[size] = size_results
        print(f"Size {size}: BFS - {size_results['bfs'].nodes_explored} nodes, "
              f"Dijkstra - {size_results['dijkstra'].nodes_explored} nodes")
    
    # Create efficiency plot
    fig = visualizer.plot_efficiency_analysis(graph_sizes, results)
    plt.show()


def example_different_graph_types():
    """Test different graph types."""
    print("\n=== Different Graph Types Example ===")
    
    graph_types = {
        'Random': GraphGenerator.create_random_graph(25, 0.3),
        'Grid': GraphGenerator.create_grid_graph(5, 5),
        'Tree': GraphGenerator.create_tree_graph(25, 2),
        'Complete': GraphGenerator.create_complete_graph(25)
    }
    
    for graph_name, graph in graph_types.items():
        print(f"\n{graph_name} Graph:")
        print(f"  Nodes: {len(graph.get_nodes())}")
        
        # Calculate edges
        edges = sum(len(list(graph.get_neighbors(n))) for n in graph.get_nodes()) // 2
        print(f"  Edges: {edges}")
        
        # Run search
        search = BiDirectionalSearch(graph)
        nodes = list(graph.get_nodes())
        result = search.find_path(nodes[0], nodes[-1], strategy="bfs")
        
        if result.success:
            print(f"  Path found: {len(result.path)} nodes")
            print(f"  Cost: {result.path_cost:.2f}")
            print(f"  Nodes explored: {result.nodes_explored}")
        else:
            print("  No path found")


def main():
    """Run all examples."""
    print("Bi-Directional Search Laboratory - Basic Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_search()
        example_strategy_comparison()
        example_different_graph_types()
        
        # Uncomment to see visualizations
        # example_visualization()
        # example_performance_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
