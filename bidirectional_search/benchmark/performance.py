"""
Performance benchmarking utilities for bi-directional search algorithms.
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.search import BiDirectionalSearch
from ..core.graph import Graph
from ..utils.generators import GraphGenerator
from ..utils.metrics import PerformanceMetrics


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.results = {}
    
    def benchmark_strategies(self, graph: Graph, strategies: List[str],
                           test_cases: List[tuple], iterations: int = 10) -> pd.DataFrame:
        """Benchmark multiple search strategies on given test cases."""
        results = []
        
        for strategy in strategies:
            for start, goal in test_cases:
                strategy_results = []
                
                for _ in range(iterations):
                    search = BiDirectionalSearch(graph)
                    start_time = time.time()
                    result = search.search(start, goal, strategy)
                    end_time = time.time()
                    
                    strategy_results.append({
                        'strategy': strategy,
                        'start': start,
                        'goal': goal,
                        'success': result.success,
                        'time': end_time - start_time,
                        'nodes_explored': result.nodes_explored,
                        'nodes_forward': result.nodes_explored_forward,
                        'nodes_backward': result.nodes_explored_backward,
                        'path_cost': result.path_cost if result.success else float('inf'),
                        'meeting_point': result.meeting_point,
                        'forward_depth': result.forward_depth,
                        'backward_depth': result.backward_depth
                    })
                
                # Calculate statistics
                df_strategy = pd.DataFrame(strategy_results)
                summary = df_strategy.agg({
                    'time': ['mean', 'std', 'min', 'max'],
                    'nodes_explored': ['mean', 'std', 'min', 'max'],
                    'nodes_forward': ['mean'],
                    'nodes_backward': ['mean'],
                    'path_cost': ['mean'],
                    'success': ['mean']
                }).round(4)
                
                results.append({
                    'strategy': strategy,
                    'avg_time': summary.loc['mean', 'time'],
                    'std_time': summary.loc['std', 'time'],
                    'min_time': summary.loc['min', 'time'],
                    'max_time': summary.loc['max', 'time'],
                    'avg_nodes': summary.loc['mean', 'nodes_explored'],
                    'std_nodes': summary.loc['std', 'nodes_explored'],
                    'min_nodes': summary.loc['min', 'nodes_explored'],
                    'max_nodes': summary.loc['max', 'nodes_explored'],
                    'avg_forward': summary.loc['mean', 'nodes_forward'],
                    'avg_backward': summary.loc['mean', 'nodes_backward'],
                    'avg_path_cost': summary.loc['mean', 'path_cost'],
                    'success_rate': summary.loc['mean', 'success']
                })
        
        return pd.DataFrame(results)
    
    def benchmark_scalability(self, graph_sizes: List[int], strategies: List[str],
                            graph_type: str = 'random', edge_density: float = 0.3,
                            iterations: int = 5) -> pd.DataFrame:
        """Benchmark algorithm scalability across different graph sizes."""
        results = []
        
        for size in graph_sizes:
            print(f"Benchmarking graph size: {size}")
            
            # Generate graph
            if graph_type == 'random':
                graph = GraphGenerator.create_random_graph(size, edge_density)
            elif graph_type == 'grid':
                graph = GraphGenerator.create_grid_graph(int(np.sqrt(size)))
            elif graph_type == 'spatial':
                graph = GraphGenerator.create_spatial_graph(size, edge_density)
            else:
                graph = GraphGenerator.create_random_graph(size, edge_density)
            
            # Test cases
            nodes = list(graph.get_nodes())
            test_cases = [(nodes[0], nodes[-1]), 
                         (nodes[len(nodes)//4], nodes[3*len(nodes)//4])]
            
            for strategy in strategies:
                strategy_results = []
                
                for start, goal in test_cases:
                    for _ in range(iterations):
                        search = BiDirectionalSearch(graph)
                        start_time = time.time()
                        result = search.search(start, goal, strategy)
                        end_time = time.time()
                        
                        strategy_results.append({
                            'graph_size': size,
                            'strategy': strategy,
                            'time': end_time - start_time,
                            'nodes_explored': result.nodes_explored,
                            'success': result.success,
                            'path_cost': result.path_cost if result.success else float('inf')
                        })
                
                # Calculate averages
                df_strategy = pd.DataFrame(strategy_results)
                results.append({
                    'graph_size': size,
                    'strategy': strategy,
                    'avg_time': df_strategy['time'].mean(),
                    'std_time': df_strategy['time'].std(),
                    'avg_nodes': df_strategy['nodes_explored'].mean(),
                    'std_nodes': df_strategy['nodes_explored'].std(),
                    'success_rate': df_strategy['success'].mean(),
                    'avg_path_cost': df_strategy[df_strategy['path_cost'] != float('inf')]['path_cost'].mean() if any(df_strategy['path_cost'] != float('inf')) else float('inf')
                })
        
        return pd.DataFrame(results)
    
    def benchmark_graph_types(self, graph_types: List[str], num_nodes: int,
                             strategies: List[str], iterations: int = 10) -> pd.DataFrame:
        """Benchmark across different graph types."""
        results = []
        
        for graph_type in graph_types:
            print(f"Benchmarking {graph_type} graph")
            
            # Generate graph
            if graph_type == 'random':
                graph = GraphGenerator.create_random_graph(num_nodes, 0.3)
            elif graph_type == 'grid':
                graph = GraphGenerator.create_grid_graph(int(np.sqrt(num_nodes)))
            elif graph_type == 'spatial':
                graph = GraphGenerator.create_spatial_graph(num_nodes, 0.3)
            elif graph_type == 'complete':
                graph = GraphGenerator.create_complete_graph(num_nodes)
            elif graph_type == 'tree':
                graph = GraphGenerator.create_tree_graph(num_nodes, 2)
            elif graph_type == 'scale_free':
                graph = GraphGenerator.create_scale_free_network(num_nodes, 2)
            elif graph_type == 'small_world':
                graph = GraphGenerator.create_small_world_network(num_nodes, 4, 0.3)
            else:
                continue
            
            # Test cases
            nodes = list(graph.get_nodes())
            test_cases = [(nodes[0], nodes[-1]), 
                         (nodes[len(nodes)//3], nodes[2*len(nodes)//3])]
            
            for strategy in strategies:
                strategy_results = []
                
                for start, goal in test_cases:
                    for _ in range(iterations):
                        search = BiDirectionalSearch(graph)
                        start_time = time.time()
                        result = search.search(start, goal, strategy)
                        end_time = time.time()
                        
                        strategy_results.append({
                            'graph_type': graph_type,
                            'strategy': strategy,
                            'time': end_time - start_time,
                            'nodes_explored': result.nodes_explored,
                            'success': result.success,
                            'path_cost': result.path_cost if result.success else float('inf')
                        })
                
                # Calculate averages
                df_strategy = pd.DataFrame(strategy_results)
                results.append({
                    'graph_type': graph_type,
                    'strategy': strategy,
                    'avg_time': df_strategy['time'].mean(),
                    'std_time': df_strategy['time'].std(),
                    'avg_nodes': df_strategy['nodes_explored'].mean(),
                    'std_nodes': df_strategy['nodes_explored'].std(),
                    'success_rate': df_strategy['success'].mean(),
                    'avg_path_cost': df_strategy[df_strategy['path_cost'] != float('inf')]['path_cost'].mean() if any(df_strategy['path_cost'] != float('inf')) else float('inf')
                })
        
        return pd.DataFrame(results)
    
    def parallel_benchmark(self, graph: Graph, strategies: List[str],
                          test_cases: List[tuple], iterations: int = 10,
                          max_workers: int = 4) -> pd.DataFrame:
        """Run benchmarks in parallel for faster execution."""
        results = []
        
        def run_single_benchmark(strategy, start, goal):
            """Run a single benchmark iteration."""
            search = BiDirectionalSearch(graph)
            start_time = time.time()
            result = search.search(start, goal, strategy)
            end_time = time.time()
            
            return {
                'strategy': strategy,
                'start': start,
                'goal': goal,
                'time': end_time - start_time,
                'nodes_explored': result.nodes_explored,
                'success': result.success,
                'path_cost': result.path_cost if result.success else float('inf')
            }
        
        # Create all tasks
        tasks = []
        for strategy in strategies:
            for start, goal in test_cases:
                for _ in range(iterations):
                    tasks.append((strategy, start, goal))
        
        # Run tasks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_benchmark, *task) for task in tasks]
            
            for future in as_completed(futures):
                results.append(future.result())
        
        return pd.DataFrame(results)
    
    def generate_benchmark_report(self, results_df: pd.DataFrame, 
                                  output_file: str = "benchmark_report.html") -> str:
        """Generate comprehensive benchmark report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bi-Directional Search Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Bi-Directional Search Performance Benchmark Report</h1>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <p>Total benchmark runs: {len(results_df)}</p>
                <p>Strategies tested: {', '.join(results_df['strategy'].unique())}</p>
                <p>Average success rate: {results_df['success'].mean():.2%}</p>
            </div>
            
            <h2>Detailed Results</h2>
            {results_df.to_html()}
            
            <h2>Performance Analysis</h2>
            <h3>Time Performance</h3>
            <p>Fastest strategy: {results_df.loc[results_df['time'].idxmin(), 'strategy']} 
               ({results_df['time'].min():.4f}s)</p>
            <p>Slowest strategy: {results_df.loc[results_df['time'].idxmax(), 'strategy']} 
               ({results_df['time'].max():.4f}s)</p>
            
            <h3>Node Exploration Efficiency</h3>
            <p>Most efficient: {results_df.loc[results_df['nodes_explored'].idxmin(), 'strategy']} 
               ({int(results_df['nodes_explored'].min())} nodes)</p>
            <p>Least efficient: {results_df.loc[results_df['nodes_explored'].idxmax(), 'strategy']} 
               ({int(results_df['nodes_explored'].max())} nodes)</p>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def plot_benchmark_results(self, results_df: pd.DataFrame, 
                              save_path: str = "benchmark_plots.png") -> plt.Figure:
        """Create visualization of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bi-Directional Search Benchmark Results', fontsize=16, fontweight='bold')
        
        # Time comparison
        sns.barplot(data=results_df, x='strategy', y='time', ax=axes[0, 0])
        axes[0, 0].set_title('Average Execution Time')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Node exploration comparison
        sns.barplot(data=results_df, x='strategy', y='nodes_explored', ax=axes[0, 1])
        axes[0, 1].set_title('Average Nodes Explored')
        axes[0, 1].set_ylabel('Number of Nodes')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Success rate
        sns.barplot(data=results_df, x='strategy', y='success', ax=axes[1, 0])
        axes[1, 0].set_title('Success Rate')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Path cost (for successful searches)
        successful_df = results_df[results_df['path_cost'] != float('inf')]
        if not successful_df.empty:
            sns.barplot(data=successful_df, x='strategy', y='path_cost', ax=axes[1, 1])
            axes[1, 1].set_title('Average Path Cost (Successful Searches)')
            axes[1, 1].set_ylabel('Path Cost')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No successful searches', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
