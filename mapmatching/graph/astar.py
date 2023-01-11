import heapq
import numpy as np
from loguru import logger
from collections import deque
from haversine import haversine, Unit

# from networkx import astar_path
# TODO: 寻找更快速的最短路算法

def calculate_nodes_dist(nodes:dict, src:int, dst:int, memo:dict={}, type='coord'):
    assert src in nodes and dst in nodes, "Check the input o and d."
    if (src, dst) in memo:
        return memo[(src, dst)]
    
    if type == 'coord':
        _src = nodes[src]
        _dst = nodes[dst]
        _len = haversine(
            (_src['y'], _src['x']), 
            (_dst['y'], _dst['x']), 
            unit=Unit.METERS
        )
    else:
        raise NotImplementedError
    
    return _len


class PathPlanning:
    def __init__(self, graph: dict, nodes: dict,
                 search_memo: dict = {}, nodes_dist_memo: dict = {},
                 max_steps: int = 2000, max_dist: int = 10000, level='debug'):

        self.graph = graph
        self.nodes = nodes
        self.search_memo = search_memo
        self.nodes_dist_memo = nodes_dist_memo
        self.max_steps = max_steps
        self.max_dist = max_dist
        self.level = level

    def check_od(self, src, dst):
        if src in self.graph and dst in self.graph:
            return True, None

        info = f"Trip ({src}, {dst})" + \
            f"{', `src` not in graph' if src not in self.graph else ', '}" + \
            f"{', `dst` not in graph' if dst not in self.graph else ''}"
        
        getattr(logger, self.level)(info)
        
        return False, {"status": 1, 'vpath': [], 'cost': np.inf} 

    def search(self, src, dst):
        return NotImplementedError
    
    def reconstruct_path(self):
        return NotImplementedError


class Astar(PathPlanning):
    def __init__(self, graph: dict, nodes: dict,
                 search_memo: dict = {}, nodes_dist_memo: dict = {}, 
                 max_steps: int = 2000, max_dist: int = 10000, level='debug'):
        super().__init__(graph, nodes, search_memo, nodes_dist_memo, max_steps, max_dist, level)
        
    def search(self, src, dst, max_steps=None, max_dist=None, weight='cost', output='epath'):
        # TODO: output epath / vpath. Refs: https://igraph.readthedocs.io/en/stable/tutorials/shortest_path_visualisation.html
        if src == dst:
            return {'status': 0, 'vpath': [src], 'cost': 0}
        
        if (src, dst) in self.search_memo:
            res = self.search_memo[(src, dst)]
            return res
        
        status, info = self.check_od(src, dst)
        if not status:
            return info

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
                
                _h = calculate_nodes_dist(self.nodes, dst, nxt, self.nodes_dist_memo)
                heapq.heappush(queue, (new_cost + _h, nxt) )
                came_from[nxt] = cur
                # came_from[nxt] = {"edge": attrs['eid'], 'node': cur}

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


class Bi_Astar(PathPlanning):
    def __init__(self, graph: dict, graph_r: dict, nodes: dict, 
                 search_memo: dict = {}, nodes_dist_memo: dict = {},
                 max_steps: int = 2000, max_dist: int = 10000, level='debug'):
        super().__init__(graph, nodes, search_memo, nodes_dist_memo, max_steps, max_dist, level)
        self.graph_r= graph_r

    def search(self, src, dst, max_steps=None, max_dist=None, level='debug'):
        status, info = self._check_od(src, dst, level)
        if not status:
            return info

        _memo = self._check_memo(src, dst)
        if _memo is not None:
            return _memo

        meet =self._searching(src, dst)
        if meet is None:
            return  {"status": 2, 'vpath': [], 'cost': np.inf} 

        path = self.extract_path(src, dst)
        cost = self.visited_backward[self.meet] + self.visited_forward[self.meet]
        res = {'status': 0, 'vpath': path, 'cost': cost}

        return res

    def _searching(self, src, dst):
        self.search_init(src, dst)

        def _helper(q1, q2):
            self.extend_queue(**q1)
            if self.meet is not None:
                return True

            self.extend_queue(**q2)
            if self.meet is not None:
                return True

            return False

        while self.queue_forward and self.queue_backward:
            if len(self.queue_forward) < len(self.queue_backward):
                if _helper(self.params_forward, self.params_backward):
                    break
            else:
                if _helper(self.params_backward, self.params_forward):
                    break

        if self.meet == -1:
            return -1

        return self.meet

    def search_init(self, src, dst):
        l0 = self.calculate_nodes_dist(src, dst)

        self.queue_forward = []
        self.parent_forward = {src: None}
        self.visited_forward = {src: 0}

        self.queue_backward = []
        self.parent_backward = {dst: None}
        self.visited_backward = {dst: 0}

        heapq.heappush(self.queue_forward, (l0, 0, src))
        heapq.heappush(self.queue_backward, (l0, 0, dst))

        self.params_forward = {
            'dst': dst,
            'queue': self.queue_forward,
            'visited': self.visited_forward,
            'opposite_visited': self.visited_backward,
            'parent': self.parent_forward,
            'graph': self.graph
        }

        self.params_backward = {
            'dst': src,
            'queue': self.queue_backward,
            'visited': self.visited_backward,
            'opposite_visited': self.visited_forward,
            'parent': self.parent_backward,
            'graph': self.graph_r
        }

        self.meet = None

    def extend_queue(self, dst, queue, visited, opposite_visited, parent, graph):
        _, dis, cur = heapq.heappop(queue)
        if cur not in graph:
            return None

        for nxt, cost in graph[cur].items():
            nxt_cost = dis + cost
            if not self.is_valid(nxt, nxt_cost, graph, visited):
                continue

            visited[nxt] = nxt_cost
            parent[nxt] = cur
            if nxt in opposite_visited:
                self.meet = nxt
                return nxt

            _h = self.calculate_nodes_dist(nxt, dst)
            heapq.heappush(queue, (nxt_cost + _h, nxt_cost, nxt))

        return None

    def is_valid(self, nxt, nxt_cost, graph, visited):
        if nxt not in graph:
            return False

        if nxt in visited and nxt_cost >= visited[nxt]:
            return False

        return True

    def calculate_nodes_dist(self, src: int, dst: int, type='coord'):
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
            if s is None:
                break
            path_fore.append(s)

        # extract path for backward part
        path_back = []
        s = self.meet

        while True:
            s = self.parent_backward[s]
            if s is None:
                break
            path_back.append(s)

        return list(reversed(path_fore)) + list(path_back)

    def _check_od(self, src, dst, level='debug'):
        if src in self.graph or dst in self.graph:
            return True, None

        info = f"Trip ({src}, {dst})" + \
            f"{', `src` not in graph' if src not in self.graph else ', '}" + \
            f"{', `dst` not in graph' if dst not in self.graph else ''}"

        getattr(logger, level)(info)

        return False, {"status": 1, 'vpath': [], 'cost': np.inf}

    def _check_memo(self, src, dst):
        if (src, dst) not in self.search_memo:
            return None

        return self.search_memo[(src, dst)]

    def plot_searching_boundary(self, path, network):
        points = set.union(set(self.visited_backward.keys()),
                           set(self.visited_forward.keys()))
        ax = network.df_nodes.loc[points].plot()

        eids = network.transform_node_seq_to_edge_seq(path)
        network.df_edges.loc[eids].plot(ax=ax, label='Path')
        network.df_nodes.loc[self.visited_backward].plot(
            ax=ax, label='Backword', color='r', alpha=.8)
        network.df_nodes.query(f"nid == {self.meet}").plot(
            ax=ax,label='Meet', color='blue', alpha=.8, marker="*", zorder=8)
        network.df_nodes.loc[self.visited_forward].plot(
            ax=ax, label='Forword', color='y', alpha=.8)

        ax.legend()


if __name__ == "__main__":
    from stmm.graph import GeoDigraph
    network = GeoDigraph()
    network.load_checkpoint(ckpt='../../data/network/Shenzhen_graph_pygeos.ckpt')
    # network.to_postgis('shenzhen')

    from tqdm import tqdm
    from stmm.utils.serialization import load_checkpoint
    astar_search_memo = load_checkpoint('../../data/debug/astar_search_memo.pkl')

    searcher = Bi_Astar(network.graph, network.graph_r, network.nodes)
    
    error_lst = []
    for (src, dst), ans in astar_search_memo.items():
        res = searcher.search(src, dst)
        cond = np.array(res['vpath']) == np.array(ans['vpath'])
        if isinstance(cond, np.ndarray):
            cond = cond.all()
        if not cond:
            # print(res['cost'] == ans['cost'], cond)
            print(f"\n\n({src}, {dst})\n\tans: {ans['vpath']}, {ans['cost']}\n\tres: {res['vpath']}, {res['cost']}")
