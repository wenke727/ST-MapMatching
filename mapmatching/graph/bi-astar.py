import heapq
import numpy as np
from loguru import logger
from collections import deque
from haversine import haversine, Unit

from mapmatching.graph import GeoDigraph


class Solution:
    def __init__(self, graph, graph_r, nodes, search_memo={}, nodes_dist_memo={}) -> None:
        self.graph = graph
        self.graph_r = graph_r
        self.nodes = nodes

        self.route_planning_memo = search_memo
        self.nodes_dist_memo = nodes_dist_memo


    def shortestPath(self, src, dst, level='debug'):
        status, info = self._check_od(src, dst, level)
        if not status:
            return info

        _memo = self._check_memo(src, dst)
        if _memo is not None:
            return _memo
        
        self.search_init(src, dst)
        res = self.searching()
        self.extract_path(src, dst)

        return res


    def search_init(self, src, dst):
        l0 = self.calculate_nodes_dist(src, dst)

        self.queue_forward = []
        self.parent_forward = {src: src}
        self.visited_forward = {src: 0}

        self.queue_backward = []
        self.parent_backward = {dst: dst}
        self.visited_backward = {dst: 0}

        heapq.heappush(self.queue_forward, (l0, 0, src))
        heapq.heappush(self.queue_backward, (l0, 0, dst))

        self.meet = None


    def searching(self):
        while self.queue_forward and self.queue_backward:
            if len(self.queue_forward) < len(self.queue_backward):
                self.extend_queue(self.queue_forward, self.visited_forward, self.visited_backward, self.parent_forward, self.graph)
                if self.meet is not None:
                    break
                    
                self.extend_queue(self.queue_backward, self.visited_backward, self.visited_forward, self.parent_backward, self.graph_r)
                if self.meet is not None:
                    break
            else:
                self.extend_queue(self.queue_backward, self.visited_backward, self.visited_forward, self.parent_backward, self.graph_r)
                if self.meet is not None:
                    break 
                             
                self.extend_queue(self.queue_forward, self.visited_forward, self.visited_backward, self.parent_forward, self.graph)
                if self.meet is not None:
                    break
        
        return -1 if self.meet is None else self.visited_backward[self.meet] + self.visited_forward[self.meet]


    def extend_queue(self, dst, queue, visited, opposite_visited, parent, graph):
        _, dis, cur = heapq.heappop(queue)
        for nxt, cost in graph[cur].items():
            if nxt not in graph:
                continue
        
            new_cost = visited[cur] + cost 
            if nxt in visited and new_cost >= visited[nxt]:
                continue

            visited[nxt] = cur
            parent[nxt] = cur

            if nxt in opposite_visited:
                self.meet = nxt
                return nxt

            _h = self.calculate_nodes_dist(nxt, dst)
            heapq.heappush(queue, (new_cost + _h, new_cost, nxt))

        return None


    def is_valid(self, x, y, grid, visited):
        n, m = len(grid), len(grid[0])

        if 0 <= x < n and 0<= y < m and not grid[x][y] and not (x,y) in visited:
            return True
        return False


    def calculate_nodes_dist(self, src:int, dst:int, type='coord'):
        assert src in self.nodes and dst in self.nodes, "Check the input o and d."
        if (src, dst) in self.nodes_dist_memo:
            return self.nodes_dist_memo[(src, dst)]
        
        if type == 'coord':
            _src = self.nodes[src]
            _dst = self.nodes[dst]
            _len = haversine(
                (_src['y'], _src['x']), 
                (_dst['y'], _dst['x']), 
                unit=Unit.METERS
            )
        else:
            raise NotImplementedError
        
        return _len


    def extract_path(self, src, dst):
        # extract path for foreward part
        path_fore = [self.meet]
        s = self.meet

        while True:
            s = self.parent_forward[s]
            path_fore.append(s)
            if s == src:
                break

        # extract path for backward part
        path_back = []
        s = self.meet

        while True:
            s = self.parent_backward[s]
            path_back.append(s)
            if s == dst:
                break

        return list(reversed(path_fore)) + list(path_back)


    def _check_od(self, src, dst, level='debug'):
        if src in self.graph or dst in self.graph:
            return True, None

        info = f"Trip ({src}, {dst})" + \
            f"{', `src` not in graph' if src not in self.graph else ', '}" + \
            f"{', `dst` not in graph' if dst not in self.graph else ''}"
        
        getattr(logger, level)(info)
        
        return False, {"status": 1, 'waypoints': [], 'cost': np.inf} 


    def _check_memo(self, src, dst):
        if (src, dst) not in self.route_planning_memo:
            return None
        
        return self.route_planning_memo[(src, dst)]

if __name__ == "__main__":
    network = GeoDigraph()
    network.load_checkpoint(ckpt='./cache/Shenzhen_graph_pygeos.ckpt')    

    searcher = Solution(graph, graph_r, nodes)
    searcher.shortestPath(grid, src, dst)
