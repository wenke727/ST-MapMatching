import heapq
import numpy as np
from loguru import logger
from haversine import haversine, Unit
from .astar import PathPlanning


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
