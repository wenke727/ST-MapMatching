import heapq
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import deque

class Node:
    """
    Define the node in the road network 
    """

    def __init__(self, id):
        self.val = id
        self.x, self.y = [float(i) for i in id.split(',')]
        self.prev = set()
        self.nxt = set()
        self.indegree = 0
        self.outdegree = 0

    def add(self, point):
        self.nxt.add(point)
        self.outdegree += 1

        point.prev.add(self)
        point.indegree += 1

    def check_0_out_more_2_in(self):
        return self.outdegree == 0 and self.indegree >= 2

    def move_nxt_to_prev(self, node):
        if node not in self.nxt:
            return False

        self.nxt.remove(node)
        self.prev.add(node)
        self.indegree += 1
        self.outdegree -= 1
        return True

    def move_prev_to_nxt(self, node):
        if node not in self.prev:
            return False

        self.prev.remove(node)
        self.nxt.add(node)
        self.indegree -= 1
        self.outdegree += 1
        return True


class Digraph:
    def __init__(self, edges=None, nodes=None, *args, **kwargs):
        """[summary]

        Args:
            edges (list, optional): Shape: (N, 2/3). Defaults to None.
            nodes (dict, optional): [description]. Defaults to None.
        """
        self.graph = {}
        self.prev  = {}
        self.edge  = {}
        self.node  = {}
        
        if edges is not None:
            self.build_graph(edges)

        if nodes is not None:
            assert isinstance(nodes, dict), "Check the Node format"
            self.node = nodes
        
        self.calculate_degree()


    def __str__(self):
        return ""


    def add_edge(self, start, end, length=None):
        for p in [start, end]:
            for g in [self.graph, self.prev]:
                if p in g:
                    continue
                g[p] = set()
            
        self.graph[start].add(end)
        self.prev[end].add(start)
        if length is not None:
            self.edge[(start, end)] = length
            
        pass


    def remove_edge(self, start, end):
        self.graph[start].remove(end)
        if len(self.graph[start]) == 0:
            del self.graph[start]
        
        self.prev[end].remove(start)
        if len(self.prev[end]) == 0:
            del self.prev[end]
        pass


    def build_graph(self, edges):
        for edge in edges:
            start, end, length = edge
            assert not(np.isnan(start) or np.isnan(end)), f"Check the input ({start}, {end})"
            
            if isinstance(start, float):
                start = int(start)
            if isinstance(end, float):
                end = int(end)
            
            self.add_edge(start, end, length)
        
        return self.graph


    def clean_empty_set(self):
        for item in [self.prev, self.graph]:
            for i in list(item.keys()):
                if len(item[i]) == 0:
                    del item[i]
        pass


    def calculate_degree(self,):
        self.clean_empty_set()
        self.degree = pd.merge(
            pd.DataFrame([[key, len(self.prev[key])]
                          for key in self.prev], columns=['pid', 'indegree']),
            pd.DataFrame([[key, len(self.graph[key])]
                          for key in self.graph], columns=['pid', 'outdegree']),
            how='outer',
            on='pid'
        ).fillna(0).astype(np.int).set_index('pid')
        
        return self.degree


    def get_origin_point(self,):
        
        return self.calculate_degree().reset_index().query( "indegree == 0 and outdegree != 0" ).pid.values


class DigraphAstar(Digraph):
    def __init__(self, edges=None, nodes=None, *args, **kwargs):
        super().__init__(edges, nodes, *args, **kwargs)
        
        self.route_planning_memo = {}


    def a_star(self, src, dst, max_query_size=2000, max_dist=10000, overwrite=False, plot=False, verbose=False):
        """Route planning by A star algs

        Args:
            origin ([type]): [description]
            dest ([type]): [description]
            verbose (bool, optional): [description]. Defaults to False.
            plot (bool, optional): [description]. Defaults to False.

        Returns:
            dict: The route planning result with path, cost and status.
            status_dict = {-1: 'unreachable'}
        """
        if not overwrite and (src, dst) in self.route_planning_memo:
            res = self.route_planning_memo[(src, dst)]
            return res
        
        if src not in self.graph or dst not in self.graph:
            print(f"Edge({src}, {dst})",
                f"{', origin not in graph' if src not in self.graph else ', '}",
                f"{', dest not in graph' if dst not in self.graph else ''}")
            
            return None

        frontier = [(0, src)]
        came_from, distance = {}, {}
        came_from[src] = None
        distance[src] = 0

        query_size = 0
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == dst or query_size > max_query_size:
                break
            
            for nxt in self.graph[cur]:
                if nxt not in self.graph:
                    continue
                
                new_cost = distance[cur] + self.edge[(cur, nxt)]
                if nxt not in distance or new_cost < distance[nxt]:
                    distance[nxt] = new_cost
                    if distance[nxt] > max_dist:
                        continue

                    heapq.heappush(frontier, (new_cost + self.cal_nodes_dis(dst, nxt), nxt) )
                    came_from[nxt] = cur
            query_size += 1

        if cur != dst:
            res = {'path': None, 'cost': np.inf, "status": -1} 
            self.route_planning_memo[(src, dst)] = res
            return res

        # reconstruct the route
        route, queue = [dst], deque([dst])
        while queue:
            node = queue.popleft()
            # assert node in came_from, f"({origin}, {dest}), way to {node}"
            if came_from[node] is None:
                continue
            route.append(came_from[node])
            queue.append(came_from[node])
        route = route[::-1]

        res = {'path':route, 'cost': distance[dst], 'status':1}
        self.route_planning_memo[(src, dst)] = res

        if plot:
            path_lst = gpd.GeoDataFrame([ { 's': route[i], 'e': route[i+1]} for i in range(len(route)-1) ])
            ax = path_lst.merge(self.df_edges, on=['s', 'e']).plot()
                    
        return res

