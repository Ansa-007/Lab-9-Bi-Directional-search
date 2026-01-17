#!/usr/bin/env python3
"""
Advanced benchmarking examples for the Bi-Directional Search Laboratory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidirectional_search import BiDirectionalSearch, GraphGenerator
from bidirectional_search.benchmark.performance import PerformanceBenchmark
from bidirectional_search.benchmark.comparison import AlgorithmComparison
from bidirectional_search.visualization.plotter import SearchVisualizer
import pandas as pd
import matplotlib.pyplot as plt


def benchmark_strategies():
    """Benchmark different search strategies."""
    print("=== Strategy Benchmarking ===")
    
    benchmark = PerformanceBenchmark()
    
    # Create test graph
    graph = GraphGenerator.create_random_graph(100, 0.3)
    nodes = list(graph.get_nodes())
    
    # Define test cases
    test_cases = [
        (nodes[0], nodes[-1]),
        (nodes[len(nodes)//4], nodes[3*len(nodes)//4]),
        (nodes[1], nodes[-2])
    ]
    
    # Benchmark strategies
    strategies = ["bfs", "dfs", "dijkstra", "astar"]
    results = benchmark.benchmark_strategies(graph, strategies, test_cases, iterations=10)
    
    print("\nBenchmark Results:")
    print(results.to_string())
    
    # Generate plots
    fig = benchmark.plot_benchmark_results(results)
    plt.show()
    
    return results


def scalability_analysis():
    """Analyze scalability across different graph sizes."""
    print("\n=== Scalability Analysis ===")
    
    benchmark = PerformanceBenchmark()
    
    # Test different graph sizes
    graph_sizes = [50, 100, 200, 300, 400, 500]
    strategies = ["bfs", "dijkstra"]
    
    results = benchmark.benchmark_scalability(
        graph_sizes, strategies, 
        graph_type='random', 
        edge_density=0.3,
        iterations=5
    )
    
    print("\nScalability Results:")
    print(results.to_string())
    
    # Create visualization
    visualizer = SearchVisualizer()
    fig = visualizer.plot_efficiency_analysis(graph_sizes, 
                                             {size: {s: r for s, r in group.groupby('strategy')} 
                                              for size, group in results.groupby('graph_size')})
    plt.show()
    
    return results


def graph_type_comparison():
    """Compare performance across different graph types."""
    print("\n=== Graph Type Comparison ===")
    
    benchmark = PerformanceBenchmark()
    
    graph_types = ['random', 'grid', 'spatial', 'tree', 'scale_free', 'small_world']
    strategies = ["bfs", "dijkstra"]
    
    results = benchmark.benchmark_graph_types(graph_types, 100, strategies, iterations=5)
    
    print("\nGraph Type Comparison Results:")
    print(results.to_string())
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time comparison
    for strategy in strategies:
        strategy_data = results[results['strategy'] == strategy]
        axes[0, 0].plot(strategy_data['graph_type'], strategy_data['avg_time'], 
                        'o-', label=strategy, linewidth=2, markersize=8)
    axes[0, 0].set_title('Average Time by Graph Type')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Node exploration comparison
    for strategy in strategies:
        strategy_data = results[results['strategy'] == strategy]
        axes[0, 1].plot(strategy_data['graph_type'], strategy_data['avg_nodes'], 
                        's-', label=strategy, linewidth=2, markersize=8)
    axes[0, 1].set_title('Average Nodes Explored by Graph Type')
    axes[0, 1].set_ylabel('Number of Nodes')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Success rate comparison
    for strategy in strategies:
        strategy_data = results[results['strategy'] == strategy]
        axes[1, 0].plot(strategy_data['graph_type'], strategy_data['success_rate'], 
                        '^-', label=strategy, linewidth=2, markersize=8)
    axes[1, 0].set_title('Success Rate by Graph Type')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Path cost comparison
    for strategy in strategies:
        strategy_data = results[results['strategy'] == strategy]
        strategy_data = strategy_data[strategy_data['avg_path_cost'] != float('inf')]
        if not strategy_data.empty:
            axes[1, 1].plot(strategy_data['graph_type'], strategy_data['avg_path_cost'], 
                            'd-', label=strategy, linewidth=2, markersize=8)
    axes[1, 1].set_title('Average Path Cost by Graph Type')
    axes[1, 1].set_ylabel('Path Cost')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results


def algorithm_comparison_analysis():
    """Advanced algorithm comparison with statistical analysis."""
    print("\n=== Advanced Algorithm Comparison ===")
    
    comparison = AlgorithmComparison()
    
    # Create test graph
    graph = GraphGenerator.create_random_graph(50, 0.3)
    nodes = list(graph.get_nodes())
    
    # Define test cases
    test_cases = [
        (nodes[0], nodes[-1]),
        (nodes[5], nodes[-5]),
        (nodes[10], nodes[-10])
    ]
    
    strategies = ["bfs", "dijkstra", "astar"]
    
    # Perform comparison
    comparison_data = comparison.compare_strategies(graph, strategies, test_cases, iterations=20)
    
    print("\nComparison Statistics:")
    for strategy, stats in comparison_data['statistics'].items():
        print(f"\n{strategy.upper()}:")
        print(f"  Average time: {stats['time_stats']['mean']:.4f}s ± {stats['time_stats']['std']:.4f}")
        print(f"  Average nodes: {int(stats['nodes_stats']['mean'])} ± {int(stats['nodes_stats']['std'])}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Efficiency ratio: {stats['efficiency_ratio']:.3f}")
    
    # Create heatmap
    fig = comparison.create_comparison_heatmap(comparison_data, metric='time')
    plt.show()
    
    # Generate report
    report_file = comparison.generate_comparison_report(comparison_data)
    print(f"\nDetailed comparison report saved to: {report_file}")
    
    return comparison_data


def unidirectional_comparison():
    """Compare bi-directional with unidirectional search."""
    print("\n=== Bi-Directional vs Unidirectional Comparison ===")
    
    comparison = AlgorithmComparison()
    
    # Create test graph
    graph = GraphGenerator.create_random_graph(100, 0.3)
    nodes = list(graph.get_nodes())
    
    test_cases = [(nodes[0], nodes[-1]), (nodes[10], nodes[-10])]
    strategies = ["bfs", "dijkstra"]
    
    # Perform comparison
    comparison_results = comparison.compare_with_unidirectional(graph, strategies, test_cases)
    
    print("\nImprovement Ratios:")
    for strategy, improvements in comparison_results['improvements'].items():
        print(f"\n{strategy.upper()}:")
        print(f"  Time improvement: {improvements['time_improvement']:.2f}x faster")
        print(f"  Node improvement: {improvements['nodes_improvement']:.2f}x fewer nodes")
    
    return comparison_results


def comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("\n=== Comprehensive Benchmark Suite ===")
    
    benchmark = PerformanceBenchmark()
    
    # Test multiple scenarios
    scenarios = [
        {'graph_type': 'random', 'sizes': [50, 100, 200], 'density': 0.3},
        {'graph_type': 'grid', 'sizes': [49, 100, 225], 'density': None},
        {'graph_type': 'spatial', 'sizes': [50, 100, 200], 'density': 0.4}
    ]
    
    all_results = []
    strategies = ["bfs", "dijkstra", "astar"]
    
    for scenario in scenarios:
        print(f"\nBenchmarking {scenario['graph_type']} graphs...")
        
        if scenario['graph_type'] == 'grid':
            # For grid graphs, sizes should be perfect squares
            sizes = [int(np.sqrt(size))**2 for size in scenario['sizes']]
        else:
            sizes = scenario['sizes']
        
        results = benchmark.benchmark_scalability(
            sizes, strategies,
            graph_type=scenario['graph_type'],
            edge_density=scenario['density'],
            iterations=3
        )
        
        results['graph_type'] = scenario['graph_type']
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    print("\nCombined Benchmark Results:")
    print(combined_results.to_string())
    
    # Generate comprehensive report
    report_file = benchmark.generate_benchmark_report(combined_results)
    print(f"\nComprehensive benchmark report saved to: {report_file}")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Benchmark Results', fontsize=16, fontweight='bold')
    
    for i, graph_type in enumerate(scenarios):
        graph_data = combined_results[combined_results['graph_type'] == graph_type['graph_type']]
        
        # Time performance
        ax = axes[0, i]
        for strategy in strategies:
            strategy_data = graph_data[graph_data['strategy'] == strategy]
            ax.plot(strategy_data['graph_size'], strategy_data['avg_time'], 
                   'o-', label=strategy, linewidth=2)
        ax.set_title(f'{graph_type["graph_type"].title()} - Time')
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Node exploration
        ax = axes[1, i]
        for strategy in strategies:
            strategy_data = graph_data[graph_data['strategy'] == strategy]
            ax.plot(strategy_data['graph_size'], strategy_data['avg_nodes'], 
                   's-', label=strategy, linewidth=2)
        ax.set_title(f'{graph_type["graph_type"].title()} - Nodes')
        ax.set_xlabel('Graph Size')
        ax.set_ylabel('Nodes Explored')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return combined_results


def main():
    """Run all advanced benchmarking examples."""
    print("Bi-Directional Search Laboratory - Advanced Benchmarking Examples")
    print("=" * 70)
    
    try:
        # Run benchmarking examples
        benchmark_strategies()
        scalability_analysis()
        graph_type_comparison()
        algorithm_comparison_analysis()
        unidirectional_comparison()
        comprehensive_benchmark()
        
        print("\n" + "=" * 70)
        print("All advanced benchmarking examples completed successfully!")
        
    except Exception as e:
        print(f"Error running benchmarking examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
