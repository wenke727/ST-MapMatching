import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from collections import defaultdict, deque

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


class LongestPath:
    """
    @param n: The number of nodes
    @param starts: One point of the edge
    @param ends: Another point of the edge
    @param lens: The length of the edge
    @return: Return the length of longest path on the tree.
    """
    def __init__(self, edges:pd.DataFrame, origin):
        starts, ends, lens = edges.start.values, edges.end.values, edges.length.values
        graph = self.build_graph(starts, ends, lens)
        self.graph = graph
        
        start, _, _  = self.bfs_helper(graph, origin)
        end, self.length, path = self.bfs_helper(graph, start)
        self.path = self.get_path(start, end, path)

        return
    
    def build_graph(self, starts, ends, lens):
        graph = defaultdict(list)
        for i in range(len(starts)):
            graph[starts[i]].append((starts[i], ends[i], lens[i]))
            graph[ends[i]].append((ends[i], starts[i], lens[i]))
            
        return graph

    def bfs_helper(self, graph, start):
        queue = deque([(start, 0)])
        path = {start: None}
        end, max_length = 0, 0
        
        while queue:
            cur, sum_length = queue.pop()
            max_length = max(max_length, sum_length)
            if max_length == sum_length:
                end = cur

            for _, nxt, edge_len in graph[cur]:
                if nxt in path:
                    continue

                path[nxt] = cur
                queue.appendleft((nxt, sum_length + edge_len))

        return end, max_length, path

    def get_path(self, start, end, visit):
        res = []
        cur = end
        while cur in visit:
            res.append(cur)
            cur = visit[cur]
        
        return res[::-1]

