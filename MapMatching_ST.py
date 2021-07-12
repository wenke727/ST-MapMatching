# %%
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from haversine import haversine, haversine_np
from collections import defaultdict, deque
from queue import PriorityQueue
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import math
import seaborn as sns
from coordTransfrom_shp import coord_transfer

import warnings
warnings.filterwarnings('ignore')

ROOT_PATH = '/home/pcl/traffic/map_factory'


# %%
class Matching():
    # deprecated
    def cal_dis_matrix(self, df, xy_cols=['x', 'y']):
        # return kilometers
        dis_matrix = df[xy_cols].values
        return haversine_np((dis_matrix[:, 0],  dis_matrix[:, 1]),  (dis_matrix[:, np.newaxis][:, :, 0], dis_matrix[:, np.newaxis][:, :, 1]))

    def __init__(self, 
                 fn_road=os.path.join(ROOT_PATH, 'map/rsrc/futian.xml'), 
                 fn_road_info=os.path.join(ROOT_PATH, 'input/osm_road_speed.xlsx'), 
                 coord_sys='gcj'):
        # 从API下载原始的xml -> 拓扑关系
        # osm_fn = '/home/pcl/ETA/map/rsrc/futian.xml'
        # bbox = [114.02973,22.52807,114.06777,22.56342]
        # download_map(osm_fn, ','.join([str(x) for x in bbox]))
        
        self.fn_road      = fn_road
        self.coord_sys    = coord_sys
        self.fn_road_info = fn_road_info
        self.nodes, self.edges = self.get_road_network( self.fn_road, self.fn_road_info, in_sys='wgs', out_sys = self.coord_sys)


    def get_road_network(self, fn, fn_road, in_sys = 'wgs', out_sys = 'wgs' ):
        '@params: fn, fn_road'
        import json
        import xml.dom.minidom
        road_type_filter = ['motorway','motorway_link', 'primary', 'primary_link','secondary', 'secondary_link','tertiary']
        dom = xml.dom.minidom.parse(fn)
        root = dom.documentElement
        nodelist = root.getElementsByTagName('node')
        waylist = root.getElementsByTagName('way')

        # nodes
        node_dic = {}
        for node in nodelist:
            node_id = node.getAttribute('id')
            node_lat = float(node.getAttribute('lat'))
            node_lon = float(node.getAttribute('lon'))
            node_dic[int(node_id)] = (node_lat, node_lon)
        nodes = pd.DataFrame(node_dic).T.rename(columns={0:'y', 1:'x'})
        nodes = gpd.GeoDataFrame( nodes, geometry= nodes.apply(lambda i: Point(i.x, i.y), axis=1) )
        nodes = coord_transfer(nodes, in_sys, out_sys)
        nodes.loc[:,['x']], nodes.loc[:,['y']]  = nodes.geometry.x, nodes.geometry.y

        # edges
        edges = []
        for way in waylist:
            taglist = way.getElementsByTagName('tag')
            road_flag = False
            road_type = None
            for tag in taglist:
                if tag.getAttribute('k') == 'highway':
                    road_flag = True
                    road_type = tag.getAttribute('v')
                    break

            if  road_flag:
                road_id = way.getAttribute('id')
                ndlist = way.getElementsByTagName('nd')
                nds,e = [], []
                for nd in ndlist:
                    nd_id = nd.getAttribute('ref')
                    nds.append( nd_id )
                for i in range( len(nds)-1 ):
                    edges.append( [nds[i], nds[i+1], road_id, road_type] )

        edges = pd.DataFrame( edges ).rename(columns={0:'s', 1:'e',2: 'road_id', 3: 'road_type'})
        edges = edges.query( f'road_type in {road_type_filter}' )
        edges.loc[:, ['s','e']] = pd.concat((edges.s.astype(np.int64), edges.e.astype(np.int64)), axis=1)

        edges = edges.merge( nodes[['x','y']], left_on='s', right_index=True ).rename(columns={'x':'x0', 'y':'y0'}
                    ).merge( nodes[['x','y']], left_on='e', right_index=True ).rename(columns={'x':'x1', 'y':'y1'})
        edges = gpd.GeoDataFrame( edges, geometry = edges.apply( lambda i: LineString( [[i.x0, i.y0], [i.x1, i.y1]] ), axis=1 ) )
        edges.loc[:,'l'] = edges.apply(lambda i: haversine_np((i.y0, i.x0), (i.y1, i.x1))*1000, axis=1)
        # edges.drop(columns=['x0','y0','x1','y1'], inplace=True)
        
        # nodes filter
        ls = np.unique(np.hstack((edges.s.values, edges.e.values)))
        nodes = nodes.loc[ls,:]

        # fn_road = None
        if fn_road:
            road_speed = pd.read_excel(fn_road)[['road_type', 'v']]
            edges = edges.merge( road_speed, on ='road_type' )
        return nodes, edges.set_index('s')


    def download_map(self, fn, bbox, verbose=False):
        if not fn.exists():
            if verbose:
                print("Downloading {}".format(fn))
            import requests
            url = f'http://overpass-api.de/api/map?bbox={bbox}'
            r = requests.get(url, stream=True)
            with fn.open('wb') as ofile:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        ofile.write(chunk)
        print(f'osm_{fn}')


    def get_bbox(self, gdf, thres=0.1, fixed_value=None):
        [x0, y1, x1, y0] = gdf.geometry.total_bounds
        if fixed_value is None:
            [x0, y0, x1, y1] = [
                x0-(x1-x0)*thres, y1+(y1-y0)*thres, x1+(x1-x0)*thres, y0-(y1-y0)*thres]
        else:
            [x0, y0, x1, y1] = [x0 - fixed_value, y0 -
                                fixed_value, x1 + fixed_value, y1 + fixed_value]
        bbox = [round(x, 6) for x in [x0, y0, x1, y1]]
        return bbox


    def plot_road_network(self, buffer_scale=.1):
        fig, ax = plt.subplots(figsize=(16, 16))
        [x0, y0, x1, y1] = self.get_bbox(self.edges, buffer_scale=0.1)
        # map_background([x0, y0, x1, y1], ax, pic_type='y', in_sys='gcj', out_sys = self.coord_sys)
        self.nodes.plot(ax=ax)
        self.edges.plot(ax=ax, color='gray')
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        plt.title("Road network")

        pass


    def shrink_road_network(self, tList):
        # TODO change to `sindex`
        # shrink_road_network according to the boundary of trajectory with a reasonable buffer
        bb = pd.DataFrame([x.geometry.total_bounds for x in tList])
        # TODO add: set the value accoding to travel distance, (tra.time.shift(-1) - tra.time).dt.total_seconds()
        coord_gap = 0.005
        bb = [bb[0].min()-coord_gap, bb[1].min()-coord_gap,
              bb[2].max()+coord_gap, bb[3].max()+coord_gap]
        nodes_ = self.nodes.query(
            f'({bb[1]} < y < {bb[3]}) and ({bb[0]} < x < {bb[2]})')
        edges_ = self.edges.query(
            f's in {list(nodes_.index)} or e in {list(nodes_.index)} ')
        return nodes_, edges_


    def Dijkstra(self, V, E, s, dest, nodes, verbose=False, fig_plot=True):
        d = [-1 if v != s else 0 for v in V]  # 出发点到各点的已知最短路径的初始值, 到自身为0
        N = len(V)  # 顶点数
        p = [None] * N  # 出发点到各点的已知最短路径的前一个点
        S = set()  # 这个集合暂时没什么用处
        Q = set(range(N))  # 生成所有顶点的虚拟序号的集合, 以0起始

        def Extract_Min(Q):  # 将顶点集合Q中有最小d[u]值的顶点u从Q中删除并返回u，可先不管
            u_min = None
            for u in Q:
                if d[u] != -1:  # 无穷大则跳过
                    origin = u_min
                    u_min = u
                    if verbose:
                        print(f'\nu: {origin} -> {u_min}, value {d[u]}')
            for u in Q:
                if d[u] == -1:  # 无穷大则跳过d
                    continue
                if d[u_min] > d[u]:
                    u_min = u
                    if verbose:
                        print(f'\tu: -> {u_min}')

            Q.remove(u_min)
            if verbose:
                print(f'\tu_min: {u_min}, Q length: {len(Q)}')
            return u_min

        v_d = V.index(dest)  # 目的地的虚拟序号
        try:
            while Q:  # 当Q非空
                u = Extract_Min(Q)  # 将顶点集合Q中有最小d[u]值的顶点u从Q中删除并返回u
                if verbose:
                    print(f'\tExtract_Min: {u}')
                if u == v_d:
                    if verbose:
                        print('Done!!!')
                    break  # 可以在找到目的地之后立即终止，也可继续查找完所有最短路径
                S.add(u)

                # v_reached_from_u = [i for i in range(N) if dis_matrix[u][i] != -1] # u能到达的顶点, 这里没有去除在S中的顶点
                v_reached_from_u = [
                    V.index(u) for u in E.query(f's == {V[u]}').e.values]
                for v in v_reached_from_u:
                    dis = E.query(f's == {V[u]}  and e == {V[v]}').l.values[0]
                    if d[v] > d[u] + dis or d[v] == -1:  # 如果经过u再到v的距离更短
                        origin = d[v]
                        d[v] = d[u] + dis  # 用该值更新已知最短路径
                        if verbose:
                            print(
                                f'\t{v}, node({V[v]}) distance: {origin} -> {d[v]}, d[{u}]: {d[u]}')
                        p[v] = u  # 则已知到v的最短路径上的倒数第二个点为u
                # print('d=', d)

            v = v_d  # 由p列表可以回溯出最短路径
            final_route = []
            while v != V.index(s):
                final_route.insert(0, V[v])
                v = p[v]
            final_route.insert(0, V[v])
            if verbose:
                print('最终路径: ', ' -> '.join([str(x) for x in final_route]))

            if fig_plot:
                fig, ax = plt.subplots(figsize=(9, 6))
                self.edges.plot(ax=ax, color='gray', zorder=0)
                show = nodes.query(f'index in {final_route}')
                show.plot(ax=ax, zorder=1)
                bbox = self.get_bbox(show, fixed_value=0.002)
                plt.xlim(bbox[0], bbox[2])
                plt.ylim(bbox[1], bbox[3])
                ax.set_title(f'dis: {dis}')

            return final_route, d[v_d]
        except:
            return [], np.inf


    def a_star_search(self, start, goal, verbose):
        def _heuristic(a, b):
            return haversine(self.nodes.loc[a][['y', 'x']].values, self.nodes.loc[b][['y', 'x']].values)*1000

        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from, cost_so_far = {}, {}
        came_from[start] = None
        cost_so_far[start] = 0

        try:
            while not frontier.empty():
                current = frontier.get()[1]
                if current == goal:
                    break

                val = self.edges.loc[current].e
                next_points = [val] if isinstance(val, int) else val
                for next in next_points:
                    new_cost = cost_so_far[current] + self.edges.query(f's=={current} and e =={next}').l.values[0]
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        if cost_so_far[next] > 10**4:
                            continue
                        priority = new_cost + _heuristic(goal, next)
                        frontier.put((priority, next))
                        came_from[next] = current

            # reconstruct the route
            route = [goal]
            queue = deque([goal])
            while queue:
                node = queue.popleft()
                if came_from[node] is None:
                    continue
                route.append(came_from[node])
                queue.append(came_from[node])
            return route[::-1], cost_so_far[goal]
        except:
            if verbose: 
                print(f'\t({start}, {goal}): a_star_search failed!!!')
            return [], np.inf


    def find_candidate(self, c, x, y, fig_plot=True):
        '''
        识别连续道路
        input:
            c, the candidate edges
        '''
        # x, y, c = geometry.x, geometry.y, candicate
        indegrees, adjacency, roads_set, roads_N_set = defaultdict(int), defaultdict(list), defaultdict(list), defaultdict(list)
        relations = defaultdict(list)
        res = defaultdict(list)
        indegrees_0 = deque([])

        for index, item in c.iterrows():
            indegrees[item.e] += 1
            adjacency[index].append(item.e)

        for i in c.index.values:
            if not indegrees[i]:
                indegrees_0.append(i)

        # for i in indegrees_0:
        # # TODO 分叉的处理, 立交下边的点，idea: skip this point and move to the next point, example( 5 114.262867 22.584119 )
        # 还没有想好怎么处理，目前这样子也蛮好的，后续的最短路算法会自动作出判断，走那个分叉，关键是权重的设置，以及投影的距离
        while indegrees_0:
            i = indegrees_0.popleft()
            queue = deque([i])
            # roads_set[i].append(i)
            while queue:
                pre = queue.popleft()
                # # the case: two or more sub-paths
                # if len(adjacency[pre])>1:
                #     relations[pre] = adjacency[pre]
                #     for cur in adjacency[pre]:
                #         indegrees_0.append(cur)
                #         N = c.query( f's=={pre} and e=={cur}').N.values[0]
                #         roads_N_set[cur].append( N )
                #         roads_set[cur].append([pre, cur])
                #     continue

                # single-paths
                for cur in adjacency[pre]:
                    indegrees[cur] -= 1
                    if not indegrees[cur]:
                        queue.append(cur)
                        # s = roads_set[i][-1] if len(roads_set[i])>0 else pre
                        N = c.query(f's=={pre} and e=={cur}').N.values[0]
                        roads_N_set[i].append(N)
                        roads_set[i].append([pre, cur])

        # roads_set, roads_N_set
        for i in roads_N_set:
            temp = roads_N_set[i].index(max(roads_N_set[i]))
            res[i] = [roads_set[i][temp]]

        # visulization
        if fig_plot:
            fig, ax = plt.subplots(figsize=(16, 16))
            color = ['b', 'g', 'black', 'gray']
            self.edges.plot(ax=ax, color='gray', zorder=0, linewidth=0.5)
            for i, begin in enumerate(roads_set):
                ps = roads_set[begin]
                plt.scatter(self.nodes.loc[begin].geometry.x, self.nodes.loc[begin].geometry.y, color=color[i % len(
                    color)], marker='o', alpha=0.5, zorder=4, s=80, label=f'{ begin } {res[begin][0]}', )
                c.merge(pd.DataFrame(ps).rename(columns={0: 's', 1: 'e'}), on=['s', 'e']).plot(ax=ax, color=color[i % len(color)], zorder=2, linestyle='--')
                c.merge(pd.DataFrame(res[begin]).rename(columns={0: 's', 1: 'e'}), on=['s', 'e']).plot(ax=ax, color=color[i % len(color)], zorder=3, linewidth=10)
            [x0, y0, x1, y1] = self.get_bbox(c, thres=0.5)
            # map_background([x0, y0, x1, y1], ax, pic_type='y', in_sys='gcj', out_sys = self.coord_sys)
            plt.scatter(x, y, marker='*', color='r', zorder=5,
                        s=200, alpha=0.5, label='GPS')
            plt.legend(loc='best', prop={'size': 15})
            plt.xlim(x0, x1)
            plt.ylim(y0, y1)
        temp  = pd.DataFrame([res[i][0] for i in res]).rename(columns={0: 's', 1: 'e'})

        try:
            c.merge(pd.DataFrame([res[i][0] for i in res]).rename(columns={0: 's', 1: 'e'}),  on=['s', 'e']).sort_values(by='N', ascending=False)
        except:
            print( f'ERROR: find_candidate {res}, len {len(res)}' )
        return c.merge(pd.DataFrame([res[i][0] for i in res]).rename(columns={0: 's', 1: 'e'}),  on=['s', 'e']).sort_values(by='N', ascending=False)


    def get_candidates(self, geometry, point_thres=0.002, dis_thres=200, std_deviation=20, fig_plot=True):
        '''
        params:nodes, edges
        default: point_thres, 220m; std_deviation, 20
        '''
        # point_thres = 0.002; dis_thres = 200; std_deviation = 20; fig_plot=True
        while True:
            ids = list(self.nodes.query(f'{geometry.x-point_thres} < x < {geometry.x+point_thres} and {geometry.y-point_thres} < y < {geometry.y+point_thres}').index)
            candicate = self.edges.query(f's in {ids} or e in {ids}')
            if candicate.shape[0] > 0:
                break
            if point_thres>0.02:
                print( 'point is a outliar, check input' )
                return None
            point_thres += 0.001

        candicate.loc[:, 'dis'] = candicate.distance(geometry) * 110*1000
        candicate.query(f'dis<{dis_thres}', inplace=True)
        # candicate = candicate[candicate.dis<dis_thres]
        
        N = 1/(np.sqrt(2*np.pi) * std_deviation) * \
            np.exp(-(np.power(candicate.dis, 2) / (2*np.power(std_deviation, 2))))
        candicate.loc[:, 'N'] = N
        candicate = self.find_candidate(candicate, geometry.x, geometry.y, fig_plot)
        candicate.loc[:, 'relative_dis'] = self.getRelativeDisFootPoint( geometry, candicate)
        return candicate.sort_values(by="N", ascending=False)


    def cos_similarity(self, path_, v_cal=30):
        # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
        seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
        v_roads = pd.DataFrame(seg, columns=['s', 'e']).merge(
            self.edges,  on=['s', 'e']).v.values
        num = np.sum(v_roads.T * v_cal)
        denom = np.linalg.norm(v_roads) * \
            np.linalg.norm([v_cal for x in v_roads])
        cos = num / denom  # 余弦值
        return cos


    def getRelativeDisFootPoint(self, point, line):
        '''
        @point, geometry, line dataframe; wgs84
        空间点到直线垂足坐标的解算及C++实现
        Ref； https://blog.csdn.net/a_222850215/article/details/79802341
        '''
        x0 = point.x
        y0 = point.y

        x1 = line.x0
        y1 = line.y0

        x2 = line.x1
        y2 = line.y1

        k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / \
            ((x2 - x1) ** 2 + (y2 - y1) ** 2)*1.0*line.l
        return k


    def find_matched_sequence(self, tList, speed=None, verbose=True, method='a_star'):
        # params: tList
        parent = defaultdict(int)
        paths = defaultdict(list)

        for i in range(1, len(tList)):
            if verbose:
                print(f'\nfind_matched_sequence total level { len(tList) }, processing level { i } ({ len(tList[i-1])*len(tList[i]) } = { len(tList[i-1]) }*{ len(tList[i]) })')
            prob = defaultdict(float)
            s_set = tList[i-1].s.values

            for end, item in tList[i].iterrows():
                max_pro = -np.inf
                for start_index, start in enumerate(s_set):
                    if start == item.s:
                        return []
                    if method == 'a_star':
                        shortest_path, shortest_path_dis = self.a_star_search(start=start, goal=item.s, verbose = verbose)
                    else:
                        shortest_path, shortest_path_dis = self.Dijkstra(s=start, dest=item.s, V=list(
                            self.nodes.index.unique()), E=self.edges, nodes=self.nodes, verbose=False, fig_plot=False)
                    # s0 =  tList[i-1].iloc[start_index].relative_dis; # shortest_path_dis += item.relative_dis - s0

                    # start point to start point
                    paths[start, item.s] = shortest_path
                    l2_dis = haversine(
                        self.nodes.loc[start, ['y', 'x']].values, self.nodes.loc[item.s, ['y', 'x']].values)*1000
                    # TODO the distance should subtract the offsets
                    dis_factor = (
                        l2_dis/shortest_path_dis) if shortest_path_dis > 0 else 0
                    speed_factor = 1 if speed is None else self.cos_similarity(
                        shortest_path, speed.iloc[i-1])
                    factor = item.N * dis_factor * speed_factor
                    trans_prob = tList[i-1].iloc[start_index].N + factor
                    if verbose:
                        print(
                            f'\t ({start}, {item.s}): {item.N:.3f}->{trans_prob:.3f}, {tList[i-1].iloc[start_index].N:.3f} + { factor:.3f} ({item.N:.3f}*{dis_factor:.2f}*{speed_factor:.2f}), [{l2_dis:.2f}, {shortest_path_dis:.2f}]')
                    if trans_prob <= max_pro or shortest_path == np.inf:
                        continue
                    max_pro = trans_prob
                    parent[item.s] = start
                    prob[item.s] = max_pro
                tList[i].loc[end, 'N'] = max_pro
                # item.N = max_pro # BUG 不能改变tList中的数值
                # print( f'\t\tupdate item.N {max_pro:.3f}, {tList[i].loc[end].N:.3f}' )

        prob_max, end_id = -np.inf, None
        for i in prob:
            if prob[i] > prob_max:
                prob_max = prob[i]
                end_id = i

        sequence, queue = [], deque([end_id])
        count = 0
        while queue:
            n = queue.popleft()
            if parent[n]:
                sequence.append(parent[n])
                queue.append(parent[n])
            count += 1
            if count > 100:  # TODO 因为算法的原因，陷入了死循环
                print( 'Error: 死循环' )
                return []
            pass

        sequence = sequence[::-1]+[end_id]
        path = []
        for i in range(len(sequence)-1):
            path += paths[sequence[i], sequence[i+1]]
        if verbose:
            print( f'path: {path}' )
        return path


    def trajectory_segmentation(self, tra, degree_threshold=5, verbose = False):
        # 线段降低维度
        # TODO shrink
        coords = []
        traj_lst = np.array(tra[['x', 'y']].values.tolist())
        coords.append(traj_lst)
        for traj_index, traj in enumerate(coords):
            hold_index_lst = []
            previous_azimuth = 1000

            for point_index, point in enumerate(traj[:-1]):
                next_point = traj[point_index + 1]
                diff_vector = next_point - point
                azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

                if abs(azimuth - previous_azimuth) > degree_threshold:
                    hold_index_lst.append(point_index)
                    previous_azimuth = azimuth
                    
            # Last point of trajectory is always added
            hold_index_lst.append(traj.shape[0] - 1)

            coords[traj_index] = traj[hold_index_lst, :]
            if verbose: 
                print(hold_index_lst)
            temp = tra.iloc[hold_index_lst, :]
        
        return temp


    def visual_map(self, tra, path=None):
        fig, ax = plt.subplots(figsize=(12, 12))
        try:
            line = LineString([self.nodes.loc[x, ['x', 'y']].values for x in path])
            ax.plot(line.xy[0], line.xy[1], alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, color='red')
        except:
            pass
        
        self.edges.plot(ax=ax, color='blue', zorder=0, linewidth=1)
        self.nodes.plot(ax=ax, color='green', zorder=1, markersize=3)
        tra.plot(ax=ax, color='red')
        [x0, y0, x1, y1] = self.get_bbox(tra, thres=0.5)
        # map_background([x0, y0, x1, y1], ax, pic_type='y', in_sys='gcj', out_sys = self.coord_sys)
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        
        return


    def ST_Matching(self, track, coord_thre=0.001, dis_thres=30, method='a_star', fig_plot_candidate=False, search_detail = False):
        '''
        params: tra(geometry, datetime), coord_thre = 0.005 # 约550m
        '''
        tra = self.trajectory_segmentation(track)
        tList, x_pre, y_pre = [], 0, 0
        con = []

        for i, item in tra.iterrows():  # TODO if current == goal: break
            x, y = item.geometry.x, item.geometry.y

            candicate = self.get_candidates(item.geometry, point_thres=coord_thre, dis_thres = dis_thres, std_deviation=20, fig_plot=fig_plot_candidate)
            tList.append(candicate)
            x_pre, y_pre = item.geometry.x, item.geometry.y
            con.append(i)

        temp = tra.loc[con]
        v = haversine_np((temp.y, temp.x), (temp.y.shift(-1), temp.x.shift(-1))) * \
            1000 / (temp.time.shift(-1) - temp.time).dt.total_seconds()

        return tList, self.find_matched_sequence(tList, method=method, verbose = search_detail)


if __name__ == '__main__':

    '''
    futian with own road dataset
    
    PROBLEM: 1. find_candidate的逻辑目前无法处理道路双向同线的情况 <- 想法：先匹配道路，再上路径匹配算法
    '''
    # tra = gpd.read_file(os.path.dirname(__file__) + '/rsrc/tra.shp', encoding='utf-8')
    # tra.time = pd.to_datetime(tra['time'], format='%Y-%m-%d %H:%M:%S')
    # tra = coord_transfer( tra, in_sys = 'wgs', out_sys = 'gcj' )
    
    # mm = Matching(fn_road = os.path.dirname(__file__) + "/rsrc/futian.xml", coord_sys='gcj')
    # mm.visual_map(tra)
    # # mm.get_candidates(tra.iloc[3].geometry, point_thres=0.002, dis_thres=200, std_deviation=20, fig_plot=True)

    # tList, path = mm.ST_Matching( tra.loc[[0,2,4,5]], coord_thre=0.001, fig_plot_candidate=True)
    # mm.visual_map(tra, path)

    # # test for candidates
    # mm.get_candidates(tra.iloc[0].geometry, point_thres=0.02, dis_thres=200, std_deviation=20, fig_plot=True)


    # Shenzhen North Station sample
    tra = gpd.read_file( os.path.dirname(__file__) + '/rsrc/tra_szn_test.geojson' )
    tra['time'] = tra.t
    tra.loc[:, 'time'] = pd.to_datetime(tra['time'], format='%Y-%m-%dT%H:%M:%S')
   
    mm = Matching(fn_road = "./input/szn_road_network_0.2.xml", coord_sys='gcj')
    tList, path = mm.ST_Matching( tra, coord_thre=0.001, fig_plot_candidate=True)
    mm.visual_map(tra, path)

    pass

#%%
