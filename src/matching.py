#%%
import os
import sys
import copy
import heapq
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from shapely.geometry import Point, LineString, box, point
from haversine import haversine, haversine_np, Unit
from coords.coordTransfrom_shp import coord_transfer

from utils.classes import Digraph
from utils.geo_plot_helper import map_visualize
from utils.df_helper import gdf_to_geojson, df_query, linestring_length, coords_pair_dist
from utils.interval_helper import merge_intervals
from utils.foot_helper import cal_foot_point_on_line

DIS_FACTOR=1/110/1000
PATH_MEMO = {}

#%%
""" Graph class """
class Digraph_OSM(Digraph):
    """
    # TODO 
        1) download net from the internet;
        2) one way
    """ 
    def __init__(self, 
                 xml_fn = '../input/futian.xml', 
                 road_info_fn='../input/osm_road_speed.xlsx', 
                 combine_link=True,
                 *args, **kwargs):
        self.df_nodes, self.df_edges = self.get_road_network(xml_fn, road_info_fn)
        self.node_dis_memo = {}
        self.route_planning_memo = {}
        super().__init__(self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)
        
        if combine_link:
            self.df_edges = self.combine_rids()
            self.df_edges.reset_index(drop=True, inplace=True)
            self.df_edges.set_crs('EPSG:4326', inplace=True)
            self.df_nodes.set_crs('EPSG:4326', inplace=True)
            # self.df_nodes = self.df_nodes.loc[ np.unique(np.hstack((self.df_edges.s.values, self.df_edges.e.values))),:]
            super().__init__(self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)
            

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


    def get_road_network(self, 
                         fn, 
                         fn_road, 
                         in_sys='wgs',
                         out_sys='wgs', 
                         signals=True,
                         road_type_filter=['motorway','motorway_link', 'primary', 'primary_link','secondary', 'secondary_link','tertiary'] ):
        import xml
        dom      = xml.dom.minidom.parse(fn)
        root     = dom.documentElement
        nodelist = root.getElementsByTagName('node')
        waylist  = root.getElementsByTagName('way')

        # nodes
        node_lst = []
        for node in tqdm(nodelist, 'Parse nodes: '):
            pid = node.getAttribute('id')
            taglist = node.getElementsByTagName('tag')
            info = {'pid': pid,
                    'y':float(node.getAttribute('lat')), 
                    'x':float(node.getAttribute('lon'))}
            
            for tag in taglist:
                if tag.getAttribute('k') == 'traffic_signals':
                    info['traffic_signals'] = tag.getAttribute('v')
            
            node_lst.append(info)
                    
        nodes = pd.DataFrame(node_lst)
        nodes = gpd.GeoDataFrame( nodes, geometry= nodes.apply(lambda i: Point(i.x, i.y), axis=1) )
        nodes = coord_transfer(nodes, in_sys, out_sys)
        nodes.loc[:, 'pid'] = nodes.pid.astype(np.int)
        nodes.loc[:,['x']], nodes.loc[:,['y']] = nodes.geometry.x, nodes.geometry.y
        nodes.set_index('pid', inplace=True)

        # traffic_signals
        self.traffic_signals = nodes[~nodes.traffic_signals.isna()].index.unique()
        
        # edges
        edges = []
        for way in tqdm(waylist, 'Parse ways: '):
            taglist = way.getElementsByTagName('tag')
            info = { tag.getAttribute('k'): tag.getAttribute('v') for tag in taglist }
            road_flag = False
            road_type = None

            if 'highway' in info:
                road_flag = True
                road_type = info['highway']

            if 'highway' in info and road_type in road_type_filter:
                info['rid'] = way.getAttribute('id')
                ndlist  = way.getElementsByTagName('nd')
                nds, e  = [], []
                # TODO: 针对单向路段，需要增加其反向的情况
                for nd in ndlist:
                    nd_id = nd.getAttribute('ref')
                    nds.append( nd_id )
                for i in range( len(nds)-1 ):
                    edges.append( { 'order': i, 's':nds[i], 'e':nds[i+1], 'road_type': road_type, **info} )

        edges = pd.DataFrame( edges )
        edges.loc[:, 'rid'] = edges.loc[:, 'rid'].astype(np.int)
        edges.loc[:, ['s','e']] = pd.concat((edges.s.astype(np.int64), edges.e.astype(np.int64)), axis=1)

        edges = edges.merge( nodes[['x','y']], left_on='s', right_index=True ).rename(columns={'x':'x0', 'y':'y0'}) \
                    .merge( nodes[['x','y']], left_on='e', right_index=True ).rename(columns={'x':'x1', 'y':'y1'})
        edges = gpd.GeoDataFrame( edges, geometry = edges.apply( lambda i: LineString( [[i.x0, i.y0], [i.x1, i.y1]] ), axis=1 ) )
        edges.loc[:, 'dist'] = edges.apply(lambda i: haversine_np((i.y0, i.x0), (i.y1, i.x1))*1000, axis=1)

        keep_cols = [ 'name', 'rid', 'order', 'road_type', 'lanes', 's', 'e',  'dist', 'oneway', 'maxspeed', 'geometry']
        edges = edges[keep_cols]
        
        # nodes filter
        ls = np.unique(np.hstack((edges.s.values, edges.e.values)))
        nodes = nodes.loc[ls,:]

        edges.sort_values(['rid', 'order'], inplace=True)

        # fn_road = None
        if fn_road:
            road_speed = pd.read_excel(fn_road)[['road_type', 'v']]
            edges = edges.merge( road_speed, on ='road_type' )
        
        return nodes, edges


    def __combine_edges_helper(self, origins, result=None, pre=None, roads=None, vis=False):
        """combine segment based on the node degree

        Args:
            origins ([type]): [description]
            result (list, optional): [Collection results]. Defaults to None.
            pre ([type], optional): The previous points, the case a node with more than 2 children. Defaults to None.
            roads (gpd.Geodataframe, optional): 道路数据框，含有属性 's' 和 'e'. Defaults to None.
            vis (bool, optional): [description]. Defaults to False.
        """
        for o in origins:
            pre_node = o
            path = []
            if pre is not None:
                path = [[pre,o]]
                self.remove_edge(pre,o)

            if o not in self.graph:
                return 
            
            # case: 0 indegree, > 2 outdegree
            if len(self.graph[o]) > 1:
                o_lst = list( self.graph[o] )
                self._combine_edges_helper( o_lst, result, o, roads, vis )
                return
            
            while o in self.graph and len(self.graph[o]) == 1:
                o = list(self.graph[o])[0]
                self.remove_edge( pre_node, o )
                path.append([pre_node, o])
                pre_node = o

            if roads is not None:
                assert hasattr(roads, 's') and hasattr(roads, 'e'), "attribute is missing"
                tmp = gpd.GeoDataFrame(path, columns=['s','e']).merge( roads, on=['s','e'] )
            
                ids = []
                for i in tmp.rid.values:
                    if len(ids) == 0 or ids[-1] != i:
                        ids.append(i)
                # ids = '_'.join(map(str, ids))

                if vis: map_visualize(tmp, 's')
                if result is not None: result.append([tmp, ids ])

            else:
                if result is not None: result.append([path, []])
            
        return


    def __combine_edges(self, roads=None, vis=False):
        """roads 是一开始传入的roads的df文件

        Args:
            roads ([type], optional): [description]. Defaults to None.
            vis (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        import copy
        graph_bak = copy.deepcopy(self.graph)
        prev_back = copy.deepcopy(self.prev)
        
        result = [] # path, rid
        origins = self.get_origin_point()
        while len(origins) > 0:
            self._combine_edges_helper(origins, result, roads=roads)
            origins = self.get_origin_point()

        if roads is not None and vis:
            for i, _ in result:
                map_visualize(i, 's')
        
        # self.graph = graph_bak
        # self.prev = prev_back
        
        return result


    def get_intermediate_point(self):
        # TODO 通过出入度为1的路网，逐一去识别路段，然后更新属性
        return self.degree.query( "indegree == 1 and outdegree == 1" ).index.unique().tolist()
    
    
    @property
    def df_node_with_degree(self,):
        return self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()


    def combine_rid_links(self, rid, omit_rids, df_edges, plot=False, save_folder=None):
        """Combine OSM link.

        Args:
            rid (int): The id of link in OSM.
            omit_rids (df): Subset of df_edges, the start point shoule meet: 1) only has 1 indegree and 1 outdegree; 2) not the traffic_signals point.
            df_edges (df, optional): [description]. Defaults to net.df_edges.

        Returns:
            pd.DataFrame: The links after combination.
        
        Example:
            `new_roads = combine_rid_links(rid=25421053, omit_rids=omit_rids, plot=True, save_folder='../cache')`
        """
        new_roads = df_edges.query(f"rid == @rid").set_index('order')
        combine_orders = omit_rids.query(f"rid == @rid").order.values
        combine_seg_indxs = merge_intervals([[x-1, x] for x in combine_orders if x > 0])

        drop_index = []
        for start, end, _ in combine_seg_indxs:
            segs = new_roads.query(f"{start} <= order <= {end}")
            pids = np.append(segs.s.values, segs.iloc[-1]['e'])
            try:
                new_roads.loc[start, 'geometry'] = LineString([[self.node[p]['x'], self.node[p]['y']] for p in pids])
                new_roads.loc[start, 'dist'] = segs.dist.sum()
                new_roads.loc[start, 'e'] = segs.iloc[-1]['e']
            except:
                print(pids)
                raise

            drop_index += [ i for i in range(start+1, end+1) ]

        new_roads.drop(index=drop_index, inplace=True)
        new_roads.reset_index()

        if save_folder is not None:
            gdf_to_geojson(new_roads, os.path.join(save_folder, f"road_{rid}_after_combination.geojson"))
        
        if plot:
            new_roads.plot()
        
        return new_roads


    def combine_rids(self, ):
        omit_pids = [ x for x in self.get_intermediate_point() if x not in self.traffic_signals ]
        omit_records = self.df_edges.query( f"s in @omit_pids" )
        omit_rids = omit_records.rid.unique().tolist()
        keep_records = self.df_edges.query( f"rid not in @omit_rids" )

        res = []
        for rid in tqdm(omit_rids, 'Combine OSM links: '):
            res.append(self.combine_rid_links(rid, omit_records, self.df_edges))

        comb_rids = gpd.GeoDataFrame(pd.concat(res)).reset_index()
        comb_rids = keep_records.append(comb_rids)

        return comb_rids


    def cal_nodes_dis(self, o, d):
        assert o in self.node and d in self.node, "Check the input o and d."
        if (o,d) in self.node_dis_memo:
            return self.node_dis_memo[(o, d)]
        
        return haversine((self.node[o]['y'], self.node[o]['x']), (self.node[d]['y'], self.node[d]['x'])) * 1000


    def a_satr(self, origin, dest, verbose=False, plot=False):
        if (origin, dest) in self.route_planning_memo:
            res = self.route_planning_memo[(origin, dest)]
            return res
        
        if origin not in self.graph or dest not in self.graph:
            print("origin not in graph or dest not in graph")
            return None, np.inf

        frontier = [(0, origin)]
        came_from, distance = {}, {}
        came_from[origin] = None
        distance[origin] = 0

        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == dest:
                break

            for nxt in self.graph[cur]:
                if nxt not in self.graph:
                    continue
                
                new_cost = distance[cur] + self.edge[(cur, nxt)]
                if nxt not in distance or new_cost < distance[nxt]:
                    distance[nxt] = new_cost
                    if distance[nxt] > 10**4:
                        continue

                    heapq.heappush(frontier, (new_cost + self.cal_nodes_dis(dest, nxt), nxt) )
                    came_from[nxt] = cur

        # reconstruct the route
        route, queue = [dest], deque([dest])
        while queue:
            node = queue.popleft()
            if came_from[node] is None:
                continue
            route.append(came_from[node])
            queue.append(came_from[node])
        route = route[::-1]

        if plot:
            path_lst = gpd.GeoDataFrame([ { 's': route[i], 'e': route[i+1]} for i in range(len(route)-1) ])
            ax = path_lst.merge(self.df_edges, on=['s', 'e']).plot()
        res = {'path':route, 'cost': distance[dest]}
        self.route_planning_memo[(origin, dest)] = res
        
        return res

""" parse xml to edge and node without combiantion"""
# net = Digraph_OSM(xml_fn='../input/futian.xml', combine_link=False)
""" parse xml to edge and node with combination"""
net = Digraph_OSM(xml_fn='../input/futian.xml', combine_link=True)


#%%

""" functions """
def load_trajectory(fn = '../input/tra.shp'):
    tra = gpd.read_file(fn, encoding='utf-8')
    tra.set_crs('EPSG:4326', inplace=True)
    tra.time = pd.to_datetime(tra['time'], format='%Y-%m-%d %H:%M:%S')
    # tra = coord_transfer( tra, in_sys = 'wgs', out_sys = 'gcj' )

    return tra


def get_candidates(traj, edges, georadius=20, top_k=5, dis_factor=DIS_FACTOR, filter=True, verbose=True, plot=True):
    """Get candidates edges for traj

    Args:
        traj (geodataframe): Trajectory T = p1 -> p2 -> ... -> pn
        edges (geodataframe): The graph edge.
        georadius (int, optional): [description]. Defaults to 20.
        dis_factor (float, optional): Factor of change lonlat to meter. Defaults to 1/110/1000.
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    def _filter_candidate(df, top_k, verbose=True):
        df_candidates = copy.deepcopy(df)
        origin_size = df_candidates.shape[0]

        df_candidates_new = df_candidates.merge(net.df_edges, right_index=True, left_on='rindex')\
                        .sort_values(['pid', 'dist_to_line'], ascending=[True, True])\
                        .groupby(['pid', 'rid'])\
                        .head(1)[['pid', 'rindex', 'rid', 's', 'e','dist_to_line']]
        df_candidates_new[['pid', 'rindex', 'rid', 's', 'e','dist_to_line']].reset_index(drop=True)
        df_candidates_new = df_candidates_new.groupby('pid').head(top_k).reset_index(drop=True)
        
        if verbose:
            print(f"Shrink candidate link {origin_size} -> {df_candidates_new.shape[0]} by choose the closest link in a road")

        return df_candidates_new

    # Ref: https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.sindex.html?highlight=sindex
    georadius = georadius*dis_factor
    boxes = traj.geometry.apply(lambda i: box(i.x-georadius, i.y-georadius,i.x+georadius, i.y+georadius))
    df_candidates = boxes.apply(lambda x: edges.sindex.query(x, predicate='intersects')).explode()
    df_candidates = pd.DataFrame(df_candidates).reset_index().rename(columns={'index': 'pid', 'geometry':'rindex'})

    candidates = df_candidates.merge(traj['geometry'], left_on='pid', right_index=True)\
                              .merge(edges['geometry'], left_on='rindex', right_index=True)\
                              .rename(columns={'geometry_x': 'point_geom', 'geometry_y': 'edge_geom'})

    # set_crs: out_sys=32649;  CRS(f"EPSG:{out_sys}")
    candidates.loc[:, 'dist_to_line'] = candidates.apply(lambda x: x.point_geom.distance(x.edge_geom) / DIS_FACTOR, axis=1)


    if filter:
        candidates_filtered = _filter_candidate(candidates, top_k)
        
    if plot:
        ax = edges.loc[candidates.rindex.values].plot()
        if filter:
            edges.loc[candidates_filtered.rindex.values].plot(ax=ax, color='red')
        traj.plot(ax=ax)

    return candidates_filtered if filter else candidates.sort_values(['pid', 'dist_to_line']).groupby('pid').head(5)


def cal_observ_prob(df, std_deviation=10):
    observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)
    
    def helper(x):
        return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))
    
    df.loc[:, 'observ_prob'] = df.dist_to_line.apply( helper)
    df.loc[:, 'observ_prob'] = df.observ_prob / df.observ_prob.max()

    return df


def cal_relative_offset(node:Point, linestring:LineString, verbose=False):
    lines = [LineString((linestring.coords[i], linestring.coords[i+1])) for i in range(len(linestring.coords)-1) ]
    lines = gpd.GeoDataFrame( {'geometry': lines} )
    # lengths = [haversine(linestring.coords[i][::-1], linestring.coords[i+1][::-1]) for i in range(len(linestring.coords)-1) ]
    # lines = gpd.GeoDataFrame( {'geometry': lines, "_length": lengths} )
    lines.loc[:, 'foot_info'] = lines.apply(lambda x: cal_foot_point_on_line(node, x.geometry), axis=1)
    lines.loc[:, 'foot_factor'] = lines.loc[:, 'foot_info'].apply(lambda x: x['flag'])
    lines.sort_values('foot_factor', inplace=True)
    linestring_length(lines, add_to_att=True, key="_length")

    if lines.foot_factor.max() < 0:
        foot = lines.iloc[-1]['foot_info']['foot'][::-1]
        point = lines.iloc[-1].geometry.coords[0][::-1]

        seg_offset = - haversine(foot, point) * 1000
        prev_offeset = 0
        offset = seg_offset + prev_offeset

        if verbose: print('case 0', foot, point, seg_offset, prev_offeset, offset)
    elif lines.foot_factor.min() > 1:
        foot = lines.iloc[0]['foot_info']['foot'][::-1]
        point = lines.iloc[0].geometry.coords[-1][::-1]

        seg_offset = haversine(foot, point) * 1000
        prev_offeset = lines.query(" foot_factor > 1")['_length'].sum()
        offset = seg_offset + prev_offeset

        if verbose: print('case 1', foot, point, seg_offset, prev_offeset, offset)
    else:
        line = lines.query(" 0 < foot_factor < 1")
        assert line.shape[0] == 1, "check the input for calculate foot"
        
        foot = line.iloc[0]['foot_info']['foot'][::-1]
        point = line.iloc[0].geometry.coords[0][::-1]
        
        seg_offset = haversine(foot, point) * 1000
        prev_offeset = lines.query(" foot_factor > 1")['_length'].sum()
        offset = seg_offset + prev_offeset

        if verbose: print('case 2',foot, point, seg_offset, prev_offeset, offset)

    return offset


def linestring_combine_helper(path):
    if len(path) == 0:
        return None
    
    lst = gpd.GeoDataFrame([ {'s': path[i], 'e': path[i+1]} for i in range(len(path)-1) ])
    lines = lst.merge(net.df_edges, on=['s', 'e']).geometry.values
    points = [ l.coords[:] for l in lines ]
    
    res = []
    for lst in points:
        res += lst
    
    return LineString(res)


def cal_trans_prob(df_candidates, net):
    df_candidates.loc[:, 'offset'] = df_candidates.apply(lambda x: 
        cal_relative_offset(traj.loc[x.pid].geometry, net.df_edges.loc[x.rindex].geometry), axis=1 )
    
    tList = []
    graph_t = []
    for name, sub in df_candidates.groupby('pid'):
        tList.append(sub)

    for i in range(len(tList)-1):
        base_atts = ['pid', 'rindex','s', 'e', 'offset']
        a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
        a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1
        graph_t.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )

    graph_t = pd.concat(graph_t).reset_index(drop=True)
    graph_t.loc[:, 'shortest_path'] = graph_t.apply(lambda x: net.a_satr(x.e_0, x.s_1, plot=False), axis=1)
    graph_t.loc[:, 'dist_sht'] = graph_t.shortest_path.apply(lambda x: x['cost'])
    graph_t.loc[:, 'dist_euc'] = graph_t.apply(lambda x: coords_pair_dist(traj.loc[x.pid_0].geometry, traj.loc[x.pid_1].geometry), axis=1)
    graph_t.loc[:, 'dist_step0'] = graph_t.apply(lambda x: net.cal_nodes_dis(x.s_0, x.e_0), axis=1)
    graph_t.loc[:, 'w'] = graph_t.dist_sht + (graph_t.dist_step0 - graph_t.offset_0) + graph_t.offset_1 
    graph_t.loc[:, 'v'] = graph_t.dist_euc / graph_t.w

    # The case: o and d all on the same link
    con = graph_t.rindex_0==graph_t.rindex_1
    graph_t.loc[con, 'v'] = 1
    graph_t.loc[con, 'shortest_path'] = {'path':[], 'cost':0}

    # create geometry
    graph_t.loc[:, 'geometry'] = graph_t.shortest_path.apply(lambda x: linestring_combine_helper( x['path']))
    graph_t = gpd.GeoDataFrame(graph_t)

    # matching debug
    # matching_debug_level(tList, graph_t,1)
    # matching_debug(tList, graph_t, net)

    atts = ['pid_0', 'pid_1', 'rindex_0', 'rindex_1', 's_0', 'e_0', 's_1', 'e_1', 'observ_prob', 
            'offset_0', 'offset_1',  'dist_sht', 'dist_euc', 'dist_step0', 'w', 'v', 'geometry']
    graph_t[atts]

    return tList, graph_t


def find_matched_sequence(graph_t, df_candidates, tList, drop_dulplicates=True):
    prev_dict = {}
    f_score = {}

    for i, item in tList[0].iterrows():
        f_score[i] = item.observ_prob

    for i in range(1, len(tList)):
        for j, cur in tList[i].iterrows():
            _max = -np.inf
            for k, prev in tList[i-1].iterrows():
                alt = f_score[k] + graph_t.query(f'rindex_0=={prev.rindex} and rindex_1 == {cur.rindex}').iloc[0].v
                if alt > _max:
                    _max = alt
                    prev_dict[j] = k
                f_score[j] = _max

    rList = []
    c = max(f_score, key=f_score.get)
    for i in range(len(tList)-1, 0, -1):
        rList.append(c)
        c = prev_dict[c]
    rList.append(c)

    rList = df_candidates.loc[rList[::-1]][['s', 'e']]

    if drop_dulplicates:
        rList = rList[(rList.s != rList.s.shift(1)) | (rList.e != rList.e.shift(1))]
    
    return rList


def get_whole_path(rList, graph_t, net, plot=False):
    steps = gpd.GeoDataFrame(rList.merge(net.df_edges, on=['s','e']))

    rList.loc[:, 'nxt_s'] = rList.s.shift(-1).fillna(0).astype(np.int)
    links = rList.merge( graph_t, left_on=['e', 'nxt_s'], right_on=['e_0', 's_1'] )[['s','e', 'nxt_s', 'geometry']]

    steps.reset_index(inplace=True)
    links.reset_index(inplace=True)
    links.loc[:,'index'] = .5 + links.loc[:,'index']

    path = steps.append(links).sort_values('index')

    if plot: path.plot()
    
    return path[['index', 's', 'e', 'name', 'rid', 'lanes', 'geometry']]


def st_matching(traj, net):
    # step 1: candidate prepararation
    df_candidates = get_candidates(traj, net.df_edges, plot=True, verbose=False, filter=True)

    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates)

    # step 2.2: Spatial analysis, transmission prob
    tList, graph_t = cal_trans_prob(df_candidates, net)

    # TODO step 3: temporal analysis

    # step 4: find matched sequence
    rList = find_matched_sequence(graph_t, df_candidates, tList)
    path = get_whole_path(rList, graph_t, net, True)

    matching_debug(tList, graph_t, net)

    return path


"""" matching plot debug helper """
def matching_debug_subplot(item, net=net, ax=None, legend=True):
    """Plot the matching situation of one pair of od.

    Args:
        item (pandas.core.series.Series): One record in tList.
        net ([type], optional): [description]. Defaults to net.
        ax ([type], optional): [description]. Defaults to None.
        legend (bool, optional): [description]. Defaults to True.

    Returns:
        ax: Ax.
    
    Example:
        matching_debug_subplot(graph_t.loc[1])
    """
    i, j = item.rindex_0, item.rindex_1
    if ax is None:
        _, ax = map_visualize(traj, scale=.9, alpha=.6, color='white')
    else:
        map_visualize(traj, scale=.9, alpha=.6, color='white', ax=ax)

    # OD
    traj.loc[[item.pid_0]].plot(ax=ax, marker="*", label=f'O ({item.rindex_0})', zorder=9)
    traj.loc[[item.pid_1]].plot(ax=ax, marker="s", label=f'D ({item.rindex_1})', zorder=9)

    # path
    net.df_edges.loc[[i]].plot(ax=ax, linestyle='--', label='first step', color='green')
    graph_t.query(f"rindex_0 == {i} and rindex_1 == {j}").plot(ax=ax, color='red', label='path')
    net.df_edges.loc[[j]].plot(ax=ax, linestyle='--', label='last step', color='black')

    # aux
    ax.set_title( f"{item.rindex_0} -> {item.rindex_1}, V: {item.v:.3f}", color = 'black' if item.v < 0.7 else 'red' )
    ax.set_axis_off()
    if legend: ax.legend()
    
    return ax
    

def matching_debug_level(tList, graph_t, level, net=net):
    """PLot the matchings between levels (i, i+1)

    Args:
        tList ([type]): The candidate points.
        graph_t ([type]): [description]
        level ([type]): [description]

    Returns:
        [type]: [description]
    """
    n_rows = tList[level].shape[0]
    n_cols = tList[level+1].shape[0]

    graph_tmp = graph_t.query(f'pid_0 == {level}')

    plt.figure(figsize=(5*n_cols, 5*n_rows))
    for i in range(n_rows*n_cols):
        ax = plt.subplot(n_rows, n_cols, i + 1) 
        matching_debug_subplot(graph_tmp.iloc[i], net=net, ax=ax)

    plt.suptitle(f'Level: {level}')
    plt.tight_layout()
    plt.show()
    
    return True


def matching_debug(tList, graph_t, net):
    levels = len(tList) - 1
    for level in range(levels):
        matching_debug_level(tList, graph_t, level, net=net)
    
    return


def cos_similarity(self, path_, v_cal=30):
    # TODO 
    # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
    seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
    v_roads = pd.DataFrame(seg, columns=['s', 'e']).merge(
        self.edges,  on=['s', 'e']).v.values
    num = np.sum(v_roads.T * v_cal)
    denom = np.linalg.norm(v_roads) * \
        np.linalg.norm([v_cal for x in v_roads])
    cos = num / denom  # 余弦值
    return cos


# %%
if __name__ == '__main__':
    """ a_star 最短路算法测试 """
    # net.a_satr(1491845161, 1933843924, plot=True)

    """ test for cal_relative_offset """
    # pid, ridx = df_candidates.iloc[0]['pid'], df_candidates.iloc[0]['rindex']
    # node = traj.loc[pid].geometry
    # line = net.df_edges.loc[ridx].geometry
    # for i in [925, 926, 927, 928]:
    #     cal_relative_offset(node, net.df_edges.loc[i].geometry)

    # for i in [1103, 979, 980, 981]:
    #     cal_relative_offset(node, net.df_edges.loc[i].geometry)


    """" matching plot debug helper """
    # matching_debug(tList, graph_t)
    # matching_debug_level(tList, graph_t, 3)
    # matching_debug_level(tList, graph_t, 2)


    """ matching test"""
    traj = load_trajectory("../input/traj.geojson")
    st_matching(traj, net)

# %%
