"""
Core bi-directional search algorithm implementation.
"""

from typing import Any, List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from time import time
import heapq
from collections import deque

from .graph import Graph
from .heuristics import HeuristicFunction, ZeroHeuristic


@dataclass
class SearchResult:
    """Container for search results and statistics."""
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


class BiDirectionalSearch:
    """Professional bi-directional search implementation."""
    
    def __init__(self, graph: Graph, heuristic: Optional[HeuristicFunction] = None):
        self.graph = graph
        self.heuristic = heuristic or ZeroHeuristic()
        self.reset_statistics()
    
    def reset_statistics(self) -> None:
        """Reset search statistics."""
        self.nodes_explored = 0
        self.nodes_explored_forward = 0
        self.nodes_explored_backward = 0
        self.search_trace = []
    
    def search(self, start: Any, goal: Any, 
               strategy: str = "bfs") -> SearchResult:
        """
        Perform bi-directional search.
        
        Args:
            start: Starting node
            goal: Goal node
            strategy: Search strategy ("bfs", "dfs", "astar", "dijkstra")
        
        Returns:
            SearchResult containing path and statistics
        """
        start_time = time()
        self.reset_statistics()
        
        # Validate input
        if not self.graph.has_node(start) or not self.graph.has_node(goal):
            return SearchResult(
                path=None, path_cost=float('inf'),
                nodes_explored=0, nodes_explored_forward=0, nodes_explored_backward=0,
                time_taken=time() - start_time, success=False
            )
        
        if start == goal:
            return SearchResult(
                path=[start], path_cost=0.0,
                nodes_explored=1, nodes_explored_forward=1, nodes_explored_backward=0,
                time_taken=time() - start_time, success=True
            )
        
        # Choose search strategy
        if strategy == "bfs":
            result = self._bidirectional_bfs(start, goal)
        elif strategy == "dfs":
            result = self._bidirectional_dfs(start, goal)
        elif strategy == "astar":
            result = self._bidirectional_astar(start, goal)
        elif strategy == "dijkstra":
            result = self._bidirectional_dijkstra(start, goal)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result.time_taken = time() - start_time
        return result
    
    def _bidirectional_bfs(self, start: Any, goal: Any) -> SearchResult:
        """Bi-directional BFS implementation."""
        # Forward search
        forward_queue = deque([start])
        forward_visited = {start}
        forward_parent = {start: None}
        forward_distance = {start: 0}
        
        # Backward search
        backward_queue = deque([goal])
        backward_visited = {goal}
        backward_parent = {goal: None}
        backward_distance = {goal: 0}
        
        meeting_point = None
        best_path_cost = float('inf')
        
        while forward_queue and backward_queue:
            # Expand smaller frontier
            if len(forward_queue) <= len(backward_queue):
                meeting_point, path_cost = self._bfs_step(
                    forward_queue, forward_visited, forward_parent, forward_distance,
                    backward_visited, backward_distance, "forward"
                )
                self.nodes_explored_forward += 1
            else:
                meeting_point, path_cost = self._bfs_step(
                    backward_queue, backward_visited, backward_parent, backward_distance,
                    forward_visited, forward_distance, "backward"
                )
                self.nodes_explored_backward += 1
            
            self.nodes_explored += 1
            
            if meeting_point is not None and path_cost < best_path_cost:
                best_path_cost = path_cost
                # Reconstruct path
                path = self._reconstruct_bidirectional_path(
                    meeting_point, forward_parent, backward_parent
                )
                return SearchResult(
                    path=path, path_cost=best_path_cost,
                    nodes_explored=self.nodes_explored,
                    nodes_explored_forward=self.nodes_explored_forward,
                    nodes_explored_backward=self.nodes_explored_backward,
                    time_taken=0, success=True, meeting_point=meeting_point,
                    forward_depth=forward_distance[meeting_point],
                    backward_depth=backward_distance[meeting_point]
                )
        
        return SearchResult(
            path=None, path_cost=float('inf'),
            nodes_explored=self.nodes_explored,
            nodes_explored_forward=self.nodes_explored_forward,
            nodes_explored_backward=self.nodes_explored_backward,
            time_taken=0, success=False
        )
    
    def _bfs_step(self, queue: deque, visited: Set, parent: Dict, 
                  distance: Dict, other_visited: Set, other_distance: Dict,
                  direction: str) -> Tuple[Optional[Any], float]:
        """Single BFS step expansion."""
        if not queue:
            return None, float('inf')
        
        current = queue.popleft()
        
        # Check if current node is in other search frontier
        if current in other_visited:
            return current, distance[current] + other_distance[current]
        
        for neighbor, _ in self.graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
                
                # Check if neighbor is in other search frontier
                if neighbor in other_visited:
                    return neighbor, distance[neighbor] + other_distance[neighbor]
        
        return None, float('inf')
    
    def _bidirectional_dijkstra(self, start: Any, goal: Any) -> SearchResult:
        """Bi-directional Dijkstra's algorithm."""
        # Forward search
        forward_pq = [(0.0, start)]
        forward_distance = {start: 0.0}
        forward_parent = {start: None}
        forward_visited = set()
        
        # Backward search
        backward_pq = [(0.0, goal)]
        backward_distance = {goal: 0.0}
        backward_parent = {goal: None}
        backward_visited = set()
        
        meeting_point = None
        best_path_cost = float('inf')
        
        while forward_pq and backward_pq:
            # Expand smaller frontier
            if len(forward_pq) <= len(backward_pq):
                meeting_point, path_cost = self._dijkstra_step(
                    forward_pq, forward_distance, forward_parent, forward_visited,
                    backward_distance, "forward"
                )
                self.nodes_explored_forward += 1
            else:
                meeting_point, path_cost = self._dijkstra_step(
                    backward_pq, backward_distance, backward_parent, backward_visited,
                    forward_distance, "backward"
                )
                self.nodes_explored_backward += 1
            
            self.nodes_explored += 1
            
            if meeting_point is not None and path_cost < best_path_cost:
                best_path_cost = path_cost
                path = self._reconstruct_bidirectional_path(
                    meeting_point, forward_parent, backward_parent
                )
                return SearchResult(
                    path=path, path_cost=best_path_cost,
                    nodes_explored=self.nodes_explored,
                    nodes_explored_forward=self.nodes_explored_forward,
                    nodes_explored_backward=self.nodes_explored_backward,
                    time_taken=0, success=True, meeting_point=meeting_point,
                    forward_depth=forward_distance[meeting_point],
                    backward_depth=backward_distance[meeting_point]
                )
        
        return SearchResult(
            path=None, path_cost=float('inf'),
            nodes_explored=self.nodes_explored,
            nodes_explored_forward=self.nodes_explored_forward,
            nodes_explored_backward=self.nodes_explored_backward,
            time_taken=0, success=False
        )
    
    def _dijkstra_step(self, pq: List[Tuple[float, Any]], distance: Dict, 
                      parent: Dict, visited: Set, other_distance: Dict,
                      direction: str) -> Tuple[Optional[Any], float]:
        """Single Dijkstra step expansion."""
        if not pq:
            return None, float('inf')
        
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            return None, float('inf')
        
        visited.add(current)
        
        # Check if current node is in other search frontier
        if current in other_distance:
            return current, current_dist + other_distance[current]
        
        for neighbor, edge_weight in self.graph.get_neighbors(current):
            if neighbor not in visited:
                new_dist = current_dist + edge_weight
                if neighbor not in distance or new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    parent[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    
                    # Check if neighbor is in other search frontier
                    if neighbor in other_distance:
                        return neighbor, new_dist + other_distance[neighbor]
        
        return None, float('inf')
    
    def _bidirectional_astar(self, start: Any, goal: Any) -> SearchResult:
        """Bi-directional A* algorithm."""
        # Forward search
        forward_pq = [(self.heuristic(start, goal), 0.0, start)]
        forward_g_score = {start: 0.0}
        forward_parent = {start: None}
        forward_visited = set()
        
        # Backward search
        backward_pq = [(self.heuristic(goal, start), 0.0, goal)]
        backward_g_score = {goal: 0.0}
        backward_parent = {goal: None}
        backward_visited = set()
        
        meeting_point = None
        best_path_cost = float('inf')
        
        while forward_pq and backward_pq:
            # Expand smaller frontier
            if len(forward_pq) <= len(backward_pq):
                meeting_point, path_cost = self._astar_step(
                    forward_pq, forward_g_score, forward_parent, forward_visited,
                    backward_g_score, goal, "forward"
                )
                self.nodes_explored_forward += 1
            else:
                meeting_point, path_cost = self._astar_step(
                    backward_pq, backward_g_score, backward_parent, backward_visited,
                    forward_g_score, start, "backward"
                )
                self.nodes_explored_backward += 1
            
            self.nodes_explored += 1
            
            if meeting_point is not None and path_cost < best_path_cost:
                best_path_cost = path_cost
                path = self._reconstruct_bidirectional_path(
                    meeting_point, forward_parent, backward_parent
                )
                return SearchResult(
                    path=path, path_cost=best_path_cost,
                    nodes_explored=self.nodes_explored,
                    nodes_explored_forward=self.nodes_explored_forward,
                    nodes_explored_backward=self.nodes_explored_backward,
                    time_taken=0, success=True, meeting_point=meeting_point,
                    forward_depth=forward_g_score[meeting_point],
                    backward_depth=backward_g_score[meeting_point]
                )
        
        return SearchResult(
            path=None, path_cost=float('inf'),
            nodes_explored=self.nodes_explored,
            nodes_explored_forward=self.nodes_explored_forward,
            nodes_explored_backward=self.nodes_explored_backward,
            time_taken=0, success=False
        )
    
    def _astar_step(self, pq: List[Tuple[float, float, Any]], g_score: Dict,
                   parent: Dict, visited: Set, other_g_score: Dict,
                   goal: Any, direction: str) -> Tuple[Optional[Any], float]:
        """Single A* step expansion."""
        if not pq:
            return None, float('inf')
        
        f_score, current_g, current = heapq.heappop(pq)
        
        if current in visited:
            return None, float('inf')
        
        visited.add(current)
        
        # Check if current node is in other search frontier
        if current in other_g_score:
            return current, current_g + other_g_score[current]
        
        for neighbor, edge_weight in self.graph.get_neighbors(current):
            if neighbor not in visited:
                tentative_g = current_g + edge_weight
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    h_score = self.heuristic(neighbor, goal)
                    f_score = tentative_g + h_score
                    heapq.heappush(pq, (f_score, tentative_g, neighbor))
                    
                    # Check if neighbor is in other search frontier
                    if neighbor in other_g_score:
                        return neighbor, tentative_g + other_g_score[neighbor]
        
        return None, float('inf')
    
    def _bidirectional_dfs(self, start: Any, goal: Any) -> SearchResult:
        """Bi-directional DFS implementation."""
        # Forward search
        forward_stack = [start]
        forward_visited = {start}
        forward_parent = {start: None}
        
        # Backward search
        backward_stack = [goal]
        backward_visited = {goal}
        backward_parent = {goal: None}
        
        meeting_point = None
        
        while forward_stack and backward_stack:
            # Expand smaller frontier
            if len(forward_stack) <= len(backward_stack):
                meeting_point = self._dfs_step(
                    forward_stack, forward_visited, forward_parent,
                    backward_visited, "forward"
                )
                self.nodes_explored_forward += 1
            else:
                meeting_point = self._dfs_step(
                    backward_stack, backward_visited, backward_parent,
                    forward_visited, "backward"
                )
                self.nodes_explored_backward += 1
            
            self.nodes_explored += 1
            
            if meeting_point is not None:
                path = self._reconstruct_bidirectional_path(
                    meeting_point, forward_parent, backward_parent
                )
                return SearchResult(
                    path=path, path_cost=len(path) - 1,
                    nodes_explored=self.nodes_explored,
                    nodes_explored_forward=self.nodes_explored_forward,
                    nodes_explored_backward=self.nodes_explored_backward,
                    time_taken=0, success=True, meeting_point=meeting_point
                )
        
        return SearchResult(
            path=None, path_cost=float('inf'),
            nodes_explored=self.nodes_explored,
            nodes_explored_forward=self.nodes_explored_forward,
            nodes_explored_backward=self.nodes_explored_backward,
            time_taken=0, success=False
        )
    
    def _dfs_step(self, stack: List, visited: Set, parent: Dict,
                  other_visited: Set, direction: str) -> Optional[Any]:
        """Single DFS step expansion."""
        if not stack:
            return None
        
        current = stack.pop()
        
        # Check if current node is in other search frontier
        if current in other_visited:
            return current
        
        for neighbor, _ in self.graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)
                
                # Check if neighbor is in other search frontier
                if neighbor in other_visited:
                    return neighbor
        
        return None
    
    def _reconstruct_bidirectional_path(self, meeting_point: Any,
                                       forward_parent: Dict, backward_parent: Dict) -> List[Any]:
        """Reconstruct path from both search directions."""
        # Reconstruct forward path
        forward_path = []
        current = meeting_point
        while current is not None:
            forward_path.append(current)
            current = forward_parent[current]
        forward_path.reverse()
        
        # Reconstruct backward path
        backward_path = []
        current = backward_parent[meeting_point]
        while current is not None:
            backward_path.append(current)
            current = backward_parent[current]
        
        # Combine paths
        return forward_path + backward_path
    
    def find_path(self, start: Any, goal: Any, strategy: str = "bfs") -> SearchResult:
        """Convenience method for path finding."""
        return self.search(start, goal, strategy)
