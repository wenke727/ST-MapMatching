import heapq
import numpy as np
from loguru import logger
from collections import deque
from haversine import haversine, Unit


def calculate_nodes_dist(nodes:dict, src:int, dst:int, memo:dict={}, ll=True):
    assert src in nodes and dst in nodes, "Check the input o and d."
    if (src, dst) in memo:
        return memo[(src, dst)]
    
    if ll:
        _src = nodes[src]
        _dst = nodes[dst]
        _len = haversine(
            (_src['y'], _src['x']), 
            (_dst['y'], _dst['x']), 
            unit=Unit.METERS
        )
    else:
        _len = nodes[src]['geometry'].distance(nodes[dst]['geometry'])
    
    return _len


class PathPlanning:
    def __init__(self, graph: dict, nodes: dict,
                 search_memo: dict = {}, nodes_dist_memo: dict = {},
                 max_steps: int = 2000, max_dist: int = 10000, level='debug', ll=True):

        self.graph = graph
        self.nodes = nodes
        self.search_memo = search_memo
        self.nodes_dist_memo = nodes_dist_memo
        self.max_steps = max_steps
        self.max_dist = max_dist
        self.level = level
        self.ll = ll

    def has_edge(self, src, dst):
        if src in self.graph and dst in self.graph:
            return True

        info = f"Trip ({src}, {dst})" + \
            f"{', `src` not in graph' if src not in self.graph else ', '}" + \
            f"{', `dst` not in graph' if dst not in self.graph else ''}"
        
        getattr(logger, self.level)(info)
        
        return False

    def search(self, src, dst):
        return NotImplementedError
    
    def reconstruct_path(self):
        return NotImplementedError


class Astar(PathPlanning):
    def __init__(self, graph: dict, nodes: dict,
                 search_memo: dict = {}, nodes_dist_memo: dict = {}, 
                 max_steps: int = 2000, max_dist: int = 10000, level='debug', ll=True):
        super().__init__(graph, nodes, search_memo, nodes_dist_memo, max_steps, max_dist, level, ll)
        
    def search(self, src, dst, max_steps=None, max_dist=None, weight='cost'):
        if src == dst:
            return {'status': 0, 'vpath': [src], 'cost': 0}
        
        if (src, dst) in self.search_memo:
            res = self.search_memo[(src, dst)]
            return res
        
        if not self.has_edge(src, dst):
            return {"status": 1, 'vpath': [], 'cost': np.inf} 

        # init
        queue = [(0, src)]
        came_from = {src: None}
        distance = {src: 0}
        step_counter = 0

        max_steps = self.max_steps if max_steps is None else max_steps
        max_dist = self.max_dist if max_dist is None else max_dist

        # searching
        while queue:
            _, cur = heapq.heappop(queue)
            if cur == dst or step_counter > max_steps:
                break
            
            for nxt, attrs in self.graph[cur].items():
                if nxt not in self.graph:
                    continue
                
                new_cost = distance[cur] + attrs[weight]
                if nxt in distance and new_cost >= distance[nxt]:
                    continue

                distance[nxt] = new_cost
                if distance[nxt] > max_dist:
                    continue
                
                _h = calculate_nodes_dist(self.nodes, dst, nxt, self.nodes_dist_memo, self.ll)
                heapq.heappush(queue, (new_cost + _h, nxt) )
                came_from[nxt] = cur

            step_counter += 1

        # abnormal situation
        if cur != dst:
            res = {"status": 2, 'vpath': [], 'cost': np.inf} 
            self.search_memo[(src, dst)] = res
            return res

        # reconstruct path
        path = self.reconstruct_path(dst, came_from)
        res = {'status': 0, 'vpath': path, 'cost': distance[dst]}
        self.search_memo[(src, dst)] = res

        return res

    def reconstruct_path(self, dst, came_from):
        route, queue = [dst], deque([dst])
        while queue:
            node = queue.popleft()
            if came_from[node] is None:
                continue
            route.append(came_from[node])
            queue.append(came_from[node])
        
        return route[::-1]

