# API Reference

## Core Classes

### BiDirectionalSearch

The main class for performing bi-directional search operations.

```python
class BiDirectionalSearch:
    def __init__(self, graph: Graph, heuristic: Optional[HeuristicFunction] = None)
```

**Parameters:**
- `graph`: The graph to search on
- `heuristic`: Optional heuristic function for informed search

**Methods:**

#### search(start, goal, strategy="bfs")

Perform bi-directional search from start to goal.

```python
def search(self, start: Any, goal: Any, strategy: str = "bfs") -> SearchResult
```

**Parameters:**
- `start`: Starting node
- `goal`: Goal node
- `strategy`: Search strategy ("bfs", "dfs", "astar", "dijkstra")

**Returns:** `SearchResult` object containing search results

**Example:**
```python
search = BiDirectionalSearch(graph)
result = search.search(0, 99, strategy="bfs")
if result.success:
    print(f"Path: {result.path}")
    print(f"Cost: {result.path_cost}")
```

#### find_path(start, goal, strategy="bfs")

Convenience method for path finding.

```python
def find_path(self, start: Any, goal: Any, strategy: str = "bfs") -> SearchResult
```

### SearchResult

Data structure containing search results and statistics.

```python
@dataclass
class SearchResult:
    path: Optional[List[Any]]
    path_cost: float
    nodes_explored: int
    nodes_explored_forward: int
    nodes_explored_backward: int
    time_taken: float
    success: bool
    meeting_point: Optional[Any] = None
    forward_depth: int = 0
    backward_depth: int = 0
```

**Attributes:**
- `path`: List of nodes from start to goal (None if no path)
- `path_cost`: Total cost of the path
- `nodes_explored`: Total nodes explored by both searches
- `nodes_explored_forward`: Nodes explored by forward search
- `nodes_explored_backward`: Nodes explored by backward search
- `time_taken`: Time taken for the search in seconds
- `success`: Boolean indicating if path was found
- `meeting_point`: Node where forward and backward searches met
- `forward_depth`: Depth of meeting point from start
- `backward_depth`: Depth of meeting point from goal

## Graph Classes

### Graph

Abstract base class for graph data structures.

```python
class Graph(ABC):
    def __init__(self)
```

**Methods:**

#### add_edge(u, v, weight=None)

Add an edge between nodes u and v.

```python
def add_edge(self, u: Any, v: Any, weight: Optional[float] = None) -> None
```

#### get_neighbors(node)

Get neighbors of a node with their edge weights.

```python
def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]
```

#### add_node(node)

Add a node to the graph.

```python
def add_node(self, node: Any) -> None
```

#### has_node(node)

Check if node exists in graph.

```python
def has_node(self, node: Any) -> bool
```

#### get_nodes()

Get all nodes in the graph.

```python
def get_nodes(self) -> Set[Any]
```

### WeightedGraph

Weighted graph implementation.

```python
class WeightedGraph(Graph)
```

**Additional Methods:**

#### get_weight(u, v)

Get weight of edge between u and v.

```python
def get_weight(self, u: Any, v: Any) -> float
```

### SpatialGraph

Graph with spatial coordinates for geometric heuristics.

```python
class SpatialGraph(WeightedGraph)
```

**Additional Methods:**

#### add_node_with_coords(node, x, y)

Add a node with spatial coordinates.

```python
def add_node_with_coords(self, node: Any, x: float, y: float) -> None
```

#### get_coordinates(node)

Get coordinates of a node.

```python
def get_coordinates(self, node: Any) -> Optional[Tuple[float, float]]
```

### DirectedGraph

Directed graph implementation.

```python
class DirectedGraph(Graph)
```

**Additional Methods:**

#### get_incoming_neighbors(node)

Get incoming neighbors for directed graphs.

```python
def get_incoming_neighbors(self, node: Any) -> List[Tuple[Any, float]]
```

## Heuristic Classes

### HeuristicFunction

Abstract base class for heuristic functions.

```python
class HeuristicFunction(ABC):
    def __call__(self, node: Any, goal: Any) -> float
```

### EuclideanDistance

Euclidean distance heuristic for spatial graphs.

```python
class EuclideanDistance(HeuristicFunction):
    def __init__(self, graph: SpatialGraph)
```

### ManhattanDistance

Manhattan distance heuristic for spatial graphs.

```python
class ManhattanDistance(HeuristicFunction):
    def __init__(self, graph: SpatialGraph)
```

### ZeroHeuristic

Zero heuristic (uninformed search).

```python
class ZeroHeuristic(HeuristicFunction)
```

## Utility Classes

### GraphGenerator

Graph generation utilities.

```python
class GraphGenerator
```

**Static Methods:**

#### create_random_graph(num_nodes, edge_probability, weighted=True, seed=None)

Generate a random graph.

```python
@staticmethod
def create_random_graph(num_nodes: int, edge_probability: float, 
                       weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph
```

#### create_grid_graph(rows, cols=None, weighted=True)

Generate a grid graph.

```python
@staticmethod
def create_grid_graph(rows: int, cols: Optional[int] = None, 
                     weighted: bool = True) -> WeightedGraph
```

#### create_spatial_graph(num_nodes, edge_probability, space_size=100.0, seed=None)

Generate a spatial graph with random coordinates.

```python
@staticmethod
def create_spatial_graph(num_nodes: int, edge_probability: float,
                        space_size: float = 100.0, seed: Optional[int] = None) -> SpatialGraph
```

#### create_complete_graph(num_nodes, weighted=True, seed=None)

Generate a complete graph.

```python
@staticmethod
def create_complete_graph(num_nodes: int, weighted: bool = True,
                         seed: Optional[int] = None) -> WeightedGraph
```

#### create_tree_graph(num_nodes, branching_factor=2, weighted=True, seed=None)

Generate a tree graph.

```python
@staticmethod
def create_tree_graph(num_nodes: int, branching_factor: int = 2,
                      weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph
```

#### create_scale_free_network(num_nodes, m=2, weighted=True, seed=None)

Generate a scale-free network using Barabási-Albert model.

```python
@staticmethod
def create_scale_free_network(num_nodes: int, m: int = 2,
                              weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph
```

#### create_small_world_network(num_nodes, k=4, p=0.3, weighted=True, seed=None)

Generate a small-world network using Watts-Strogatz model.

```python
@staticmethod
def create_small_world_network(num_nodes: int, k: int = 4, p: float = 0.3,
                               weighted: bool = True, seed: Optional[int] = None) -> WeightedGraph
```

## Visualization Classes

### SearchVisualizer

Visualizer for bi-directional search process and results.

```python
class SearchVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8))
```

**Methods:**

#### plot_search_comparison(results, title)

Compare performance of different search algorithms.

```python
def plot_search_comparison(self, results: Dict[str, SearchResult], 
                          title: str = "Search Algorithm Comparison") -> plt.Figure
```

#### plot_efficiency_analysis(graph_sizes, results)

Analyze search efficiency across different graph sizes.

```python
def plot_efficiency_analysis(self, graph_sizes: List[int], 
                            results: Dict[int, Dict[str, SearchResult]]) -> plt.Figure
```

### GraphPlotter

Graph visualization utilities.

```python
class GraphPlotter:
    def __init__(self, figsize: Tuple[int, int] = (10, 8))
```

**Methods:**

#### plot_graph(graph, layout='spring', node_colors=None, edge_colors=None)

Plot graph structure.

```python
def plot_graph(self, graph: Graph, layout: str = 'spring', 
               node_colors: Optional[Dict[Any, str]] = None,
               edge_colors: Optional[Dict[Tuple[Any, Any], str]] = None) -> plt.Figure
```

#### plot_search_path(graph, result, start, goal, layout='spring')

Visualize the search path on the graph.

```python
def plot_search_path(self, graph: Graph, result: SearchResult,
                    start: Any, goal: Any, layout: str = 'spring') -> plt.Figure
```

#### plot_spatial_graph(graph, result=None, start=None, goal=None)

Plot spatial graph with coordinates.

```python
def plot_spatial_graph(self, graph: SpatialGraph, result: Optional[SearchResult] = None,
                      start: Optional[Any] = None, goal: Optional[Any] = None) -> plt.Figure
```

## Benchmarking Classes

### PerformanceBenchmark

Performance measurement utilities.

```python
class PerformanceBenchmark:
    def __init__(self)
```

**Methods:**

#### benchmark_strategies(graph, strategies, test_cases, iterations=10)

Benchmark multiple search strategies on given test cases.

```python
def benchmark_strategies(self, graph: Graph, strategies: List[str],
                       test_cases: List[tuple], iterations: int = 10) -> pd.DataFrame
```

#### benchmark_scalability(graph_sizes, strategies, graph_type='random', edge_density=0.3, iterations=5)

Benchmark algorithm scalability across different graph sizes.

```python
def benchmark_scalability(self, graph_sizes: List[int], strategies: List[str],
                         graph_type: str = 'random', edge_density: float = 0.3,
                         iterations: int = 5) -> pd.DataFrame
```

#### benchmark_graph_types(graph_types, num_nodes, strategies, iterations=10)

Benchmark across different graph types.

```python
def benchmark_graph_types(self, graph_types: List[str], num_nodes: int,
                         strategies: List[str], iterations: int = 10) -> pd.DataFrame
```

#### generate_benchmark_report(results_df, output_file="benchmark_report.html")

Generate comprehensive benchmark report.

```python
def generate_benchmark_report(self, results_df: pd.DataFrame, 
                             output_file: str = "benchmark_report.html") -> str
```

### AlgorithmComparison

Advanced algorithm comparison and analysis tools.

```python
class AlgorithmComparison:
    def __init__(self)
```

**Methods:**

#### compare_strategies(graph, strategies, test_cases, iterations=20)

Comprehensive comparison of search strategies.

```python
def compare_strategies(self, graph: Graph, strategies: List[str],
                      test_cases: List[tuple], iterations: int = 20) -> Dict[str, Any]
```

#### compare_with_unidirectional(graph, strategies, test_cases)

Compare bi-directional with unidirectional search.

```python
def compare_with_unidirectional(self, graph: Graph, strategies: List[str],
                               test_cases: List[tuple]) -> Dict[str, Any]
```

#### generate_comparison_report(comparison_data, output_file="comparison_report.html")

Generate comprehensive comparison report.

```python
def generate_comparison_report(self, comparison_data: Dict[str, Any],
                              output_file: str = "comparison_report.html") -> str
```

## Interactive Dashboard

### InteractiveDashboard

Interactive dashboard for exploring bi-directional search algorithms.

```python
class InteractiveDashboard:
    def __init__(self, port: int = 8050)
```

**Methods:**

#### run(debug=False)

Run the dashboard.

```python
def run(self, debug: bool = False) -> None
```

**Example:**
```python
dashboard = InteractiveDashboard(port=8050)
dashboard.run(debug=False)
```

## Convenience Functions

### Heuristic Factory Functions

```python
def euclidean_distance(graph: SpatialGraph) -> EuclideanDistance
def manhattan_distance(graph: SpatialGraph) -> ManhattanDistance
def zero_heuristic() -> ZeroHeuristic
def degree_heuristic(graph) -> DegreeHeuristic
```

## Error Handling

### Common Exceptions

- **ValueError**: Invalid parameters or strategy
- **KeyError**: Node not found in graph
- **TypeError**: Incorrect data types

### Example Error Handling

```python
try:
    result = search.search(start, goal, strategy="invalid")
except ValueError as e:
    print(f"Invalid strategy: {e}")
except KeyError as e:
    print(f"Node not found: {e}")
```

## Performance Tips

### 1. Graph Selection

- Use `SpatialGraph` for geometric problems with heuristics
- Use `WeightedGraph` for weighted shortest path problems
- Use `UnweightedGraph` for unweighted problems

### 2. Strategy Selection

- **BFS**: Best for unweighted graphs, guarantees shortest path
- **Dijkstra**: Best for weighted graphs, guarantees optimal path
- **A***: Best for spatial problems with good heuristics
- **DFS**: Best for memory-constrained environments

### 3. Heuristic Selection

- **Euclidean**: For continuous space problems
- **Manhattan**: For grid-based problems
- **Zero**: When no good heuristic is available

### 4. Memory Optimization

- Use appropriate graph sizes for available memory
- Clear search state between multiple searches
- Consider parallel execution for large problems

## Integration Examples

### Basic Usage

```python
from bidirectional_search import BiDirectionalSearch, GraphGenerator

# Create graph
graph = GraphGenerator.create_random_graph(100, 0.3)

# Initialize search
search = BiDirectionalSearch(graph)

# Find path
result = search.find_path(0, 99, strategy="bfs")
```

### Advanced Usage with Heuristics

```python
from bidirectional_search import BiDirectionalSearch, GraphGenerator
from bidirectional_search.core.heuristics import euclidean_distance

# Create spatial graph
graph = GraphGenerator.create_spatial_graph(100, 0.3)

# Initialize search with heuristic
heuristic = euclidean_distance(graph)
search = BiDirectionalSearch(graph, heuristic)

# Find path using A*
result = search.find_path(0, 99, strategy="astar")
```

### Benchmarking

```python
from bidirectional_search.benchmark.performance import PerformanceBenchmark

# Create benchmark
benchmark = PerformanceBenchmark()

# Run benchmarks
results = benchmark.benchmark_strategies(graph, ["bfs", "dijkstra"], test_cases)

# Generate report
benchmark.generate_benchmark_report(results)
```

### Visualization

```python
from bidirectional_search.visualization.plotter import GraphPlotter

# Create plotter
plotter = GraphPlotter()

# Plot graph with path
fig = plotter.plot_search_path(graph, result, 0, 99)
fig.show()
```

This API reference provides comprehensive documentation for all classes and methods in the Bi-Directional Search Laboratory.
