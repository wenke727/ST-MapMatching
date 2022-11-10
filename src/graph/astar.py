import heapq
import numpy as np
from loguru import logger
from collections import deque
from haversine import haversine, Unit

# TODO: 寻找更快速的最短路算法， maybe bidirection Astar

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


def _reconstruct_path(dst, came_from):
    route, queue = [dst], deque([dst])
    while queue:
        node = queue.popleft()
        if came_from[node] is None:
            continue
        route.append(came_from[node])
        queue.append(came_from[node])
    
    return route[::-1]


def a_star(graph:dict, 
           nodes:dict, 
           src:int, 
           dst:int, 
           search_memo:dict={}, 
           nodes_dist_memo:dict={}, 
           max_steps:int=2000, 
           max_dist:int=10000,
           level='debug'):
    """Route planning by A star algs

    Args:
        graph (dict): _description_
        nodes (dict): _description_
        src (int): _description_
        dst (int): _description_
        search_memo (dict, optional): _description_. Defaults to {}.
        nodes_dist_memo (dict, optional): _description_. Defaults to {}.
        max_steps (int, optional): _description_. Defaults to 2000.
        max_dist (int, optional): _description_. Defaults to 10000.

    Returns:
        dict: The route planning result with path, cost and status.
        status_dict = {-1: 'unreachable'}
    """
    if (src, dst) in search_memo:
        res = search_memo[(src, dst)]
        return res
    
    if src not in graph or dst not in graph:
        info = f"Trip ({src}, {dst})" + \
            f"{', `src` not in graph' if src not in graph else ', '}" + \
            f"{', `dst` not in graph' if dst not in graph else ''}"
        
        getattr(logger, level)(info)
        
        return {"status": 1, 'waypoints': [], 'cost': np.inf} 

    queue = [(0, src)]
    came_from = {src: None}
    distance = {src: 0}
    step_counter = 0

    while queue:
        _, cur = heapq.heappop(queue)
        if cur == dst or step_counter > max_steps:
            break
        
        for nxt, cost in graph[cur].items():
            if nxt not in graph:
                continue
            
            new_cost = distance[cur] + cost
            if nxt not in distance or new_cost < distance[nxt]:
                distance[nxt] = new_cost
                if distance[nxt] > max_dist:
                    continue
                
                _h = calculate_nodes_dist(nodes, dst, nxt, nodes_dist_memo)
                heapq.heappush(queue, (new_cost + _h, nxt) )
                came_from[nxt] = cur
        step_counter += 1

    # abnormal situation
    if cur != dst:
        res = {"status": 2, 'waypoints': [], 'cost': np.inf} 
        search_memo[(src, dst)] = res
        return res

    # reconstruct path
    path = _reconstruct_path(dst, came_from)
    res = {'status': 0, 'waypoints':path, 'cost': distance[dst]}
    search_memo[(src, dst)] = res

    return res
