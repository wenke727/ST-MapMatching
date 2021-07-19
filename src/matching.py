#%%
import os
import sys
import time
import math
import copy
import heapq
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
from collections import deque
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from shapely.geometry import Point, LineString, box

from utils.classes import Digraph
from utils.geo_plot_helper import map_visualize
from coords.coordTransfrom_shp import coord_transfer
from utils.df_helper import gdf_to_geojson, df_query, coords_pair_dist
from utils.interval_helper import merge_intervals
from utils.foot_helper import cal_foot_point_on_polyline
from utils.pickle_helper import PickleSaver
from load_step import df_path, split_line_to_points

from setting import filters as way_filters
from setting import DIS_FACTOR, DEBUG_FOLDER, SZ_BBOX, GBA_BBOX

warnings.filterwarnings('ignore')


#%%
""" Graph class """
class Digraph_OSM(Digraph):
    def __init__(self, 
                 bbox=None,
                 xml_fn='../input/futian.xml', 
                 road_info_fn='../input/osm_road_speed.xlsx', 
                 combine_link=True,
                 reverse_edge=True,
                 *args, **kwargs):
        assert not(bbox is None and xml_fn is None), "Please define one of the bbox or the xml path."
        if bbox is not None:
            xml_fn = f"../cache/osm_{'_'.join(map(str, bbox))}.xml"
            self.download_map(xml_fn, bbox, True)
        
        self.df_nodes, self.df_edges = self.get_road_network(xml_fn, road_info_fn)
        self.node_dis_memo = {}
        self.route_planning_memo = {}
        super().__init__(self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)

        self.df_edges.set_crs('EPSG:4326', inplace=True)
        self.df_nodes.set_crs('EPSG:4326', inplace=True)
                
        if combine_link:
            self.df_edges = self.combine_rids()
            self.df_edges.reset_index(drop=True, inplace=True)

        if reverse_edge:
            self.df_edges = self.add_reverse_edge(self.df_edges)
            self.df_edges.reset_index(drop=True, inplace=True)
        
        if combine_link or reverse_edge:
            # self.df_nodes = self.df_nodes.loc[ np.unique(np.hstack((self.df_edges.s.values, self.df_edges.e.values))),:]
            super().__init__(self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)
            

    def download_map(self, fn, bbox, verbose=False):
        """Download OSM map of bbox from Internet.

        Args:
            fn (function): [description]
            bbox ([type]): [description]
            verbose (bool, optional): [description]. Defaults to False.
        """
        if os.path.exists(fn):
            return

        import requests
        if verbose:
            print("Downloading {}".format(fn))
        
        if isinstance(bbox, list) or isinstance(bbox, np.array):
            bbox = ",".join(map(str, bbox))

        url = f'http://overpass-api.de/api/map?bbox={bbox}'
        r = requests.get(url, stream=True)
        with open(fn, 'wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)

        if verbose:
            print("Downloaded {} success.\n".format(fn))

        return True


    def get_road_network(self, 
                         fn, 
                         fn_road, 
                         in_sys='wgs',
                         out_sys='wgs', 
                         signals=True,
                         road_type_filter=way_filters['auto']['highway'],
                         keep_cols=['name', 'rid', 'order', 'road_type', 'lanes', 's', 'e',  'dist', 'oneway', 'maxspeed', 'geometry']
                         ):
        import xml
        dom      = xml.dom.minidom.parse(fn)
        root     = dom.documentElement
        nodelist = root.getElementsByTagName('node')
        waylist  = root.getElementsByTagName('way')

        # nodes
        nodes = []
        for node in tqdm(nodelist, 'Parse nodes: \t'):
            pid = node.getAttribute('id')
            taglist = node.getElementsByTagName('tag')
            info = {'pid': int(pid),
                    'y':float(node.getAttribute('lat')), 
                    'x':float(node.getAttribute('lon'))}
            
            for tag in taglist:
                if tag.getAttribute('k') == 'traffic_signals':
                    info['traffic_signals'] = tag.getAttribute('v')
            
            nodes.append(info)
                    
        nodes = gpd.GeoDataFrame(nodes)
        nodes.loc[:, 'geometry'] = nodes.apply(lambda i: Point(i.x, i.y), axis=1)
        nodes.set_index('pid', inplace=True)

        if in_sys != out_sys:
            nodes = coord_transfer(nodes, in_sys, out_sys)
            nodes.loc[:,['x']], nodes.loc[:,['y']] = nodes.geometry.x, nodes.geometry.y

        # traffic_signals
        self.traffic_signals = nodes[~nodes.traffic_signals.isna()].index.unique()
        
        # edges
        edges = []
        for way in tqdm(waylist, 'Parse ways: \t'):
            taglist = way.getElementsByTagName('tag')
            info = { tag.getAttribute('k'): tag.getAttribute('v') for tag in taglist }

            if 'highway' not in info or info['highway'] in road_type_filter:
                continue
            
            info['rid'] = int(way.getAttribute('id'))
            ndlist  = way.getElementsByTagName('nd')
            nds = []
            # TODO: 针对单向路段，需要增加其反向的情况
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                nds.append( nd_id )
            for i in range( len(nds)-1 ):
                edges.append( { 'order': i, 's':nds[i], 'e':nds[i+1], 'road_type': info['highway'], **info} )

        edges = pd.DataFrame( edges )
        edges.loc[:, ['s','e']] = pd.concat((edges.s.astype(np.int), edges.e.astype(np.int)), axis=1)

        edges = edges.merge( nodes[['x','y']], left_on='s', right_index=True ).rename(columns={'x':'x0', 'y':'y0'}) \
                     .merge( nodes[['x','y']], left_on='e', right_index=True ).rename(columns={'x':'x1', 'y':'y1'})
        edges = gpd.GeoDataFrame( edges, geometry = edges.apply( lambda i: LineString( [[i.x0, i.y0], [i.x1, i.y1]] ), axis=1 ) )
        edges.loc[:, 'dist'] = edges.apply(lambda i: haversine((i.y0, i.x0), (i.y1, i.x1), unit=Unit.METERS), axis=1)

        # nodes filter
        ls = np.unique(np.hstack((edges.s.values, edges.e.values)))
        nodes = nodes.loc[ls,:]

        edges.sort_values(['rid', 'order'], inplace=True)

        if fn_road:
            road_speed = pd.read_excel(fn_road)[['road_type', 'v']]
            edges = edges.merge( road_speed, on ='road_type' )
        
        return nodes, edges[keep_cols]


    def add_reverse_edge(self, df_edges):
        """Add reverse edge.

        Args:
            df_edges (gpd.GeoDataFrame): The edge file parsed from OSM.
        Check:
            rid = 34900355
            net.df_edges.query( f"rid == {rid} or rid == -{rid}" ).sort_values(['order','rid'])
        """
        def _juedge_oneway(oneway_flag):
            # https://wiki.openstreetmap.org/wiki/Key:oneway
            if oneway_flag == 'yes' or oneway_flag == '1' or oneway_flag == True:
                flag = True
            elif oneway_flag == '-1':
                flag = True
                # way.is_reversed = True
            elif oneway_flag == 'no' or oneway_flag == '0' or oneway_flag == False:
                flag = False
            elif oneway_flag in ['reversible', 'alternating']:
                # TODO: reversible, alternating: https://wiki.openstreetmap.org/wiki/Tag:oneway%3Dreversible
                flag = False
            else:
                flag = False
                # printlog(f'new maxspeed type detected at way {way.osm_way_id}, {tags["oneway"]}', 'warning')

            return flag

        df_edges.oneway = df_edges.oneway.fillna('no').apply(_juedge_oneway)
        df_edge_rev = df_edges.query('oneway == False')

        # df_edge_rev.loc[:, 'rid']      = -df_edge_rev.rid
        df_edge_rev.loc[:, 'order']    = -df_edge_rev.order - 1
        df_edge_rev.loc[:, 'geometry'] =  df_edge_rev.geometry.apply( lambda x: LineString(x.coords[::-1]) )
        df_edge_rev.rename(columns={'s':'e', 'e':'s'}, inplace=True)
        df_tmp = df_edges.append(df_edge_rev)

        return df_tmp

    def get_intermediate_point(self):
        """出入度为1的路网，逐一去识别路段，然后更新属性

        Returns:
            [type]: [description]
        """
        return self.degree.query( "indegree == 1 and outdegree == 1" ).index.unique().tolist()
    

    def combine_links_of_rid(self, rid, omit_rids, df_edges, plot=False, save_folder=None):
        """Combine OSM link.

        Args:
            rid (int): The id of link in OSM.
            omit_rids (df): Subset of df_edges, the start point shoule meet: 1) only has 1 indegree and 1 outdegree; 2) not the traffic_signals point.
            df_edges (df, optional): [description]. Defaults to net.df_edges.

        Returns:
            pd.DataFrame: The links after combination.
        
        Example:
            `new_roads = combine_links_of_rid(rid=25421053, omit_rids=omit_rids, plot=True, save_folder='../cache')`
        """
        new_roads = df_edges.query(f"rid == @rid").set_index('order')
        combine_orders = omit_rids.query(f"rid == @rid").order.values
        combine_seg_indxs = merge_intervals([[x-1, x] for x in combine_orders if x > 0])

        drop_index = []
        for start, end, _ in combine_seg_indxs:
            segs = new_roads.query(f"{start} <= order <= {end}")
            pids = np.append(segs.s.values, segs.iloc[-1]['e'])

            new_roads.loc[start, 'geometry'] = LineString([[self.node[p]['x'], self.node[p]['y']] for p in pids])
            new_roads.loc[start, 'dist'] = segs.dist.sum()
            new_roads.loc[start, 'e'] = segs.iloc[-1]['e']

            drop_index += [ i for i in range(start+1, end+1) ]

        new_roads.drop(index=drop_index, inplace=True)
        new_roads.reset_index(inplace=True)

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
        for rid in tqdm(omit_rids, 'Combine links: \t'):
            res.append(self.combine_links_of_rid(rid, omit_records, self.df_edges))

        comb_rids = gpd.GeoDataFrame(pd.concat(res))
        comb_rids = keep_records.append(comb_rids).reset_index()

        return comb_rids


    def cal_nodes_dis(self, o, d):
        assert o in self.node and d in self.node, "Check the input o and d."
        if (o, d) in self.node_dis_memo:
            return self.node_dis_memo[(o, d)]
        
        return haversine((self.node[o]['y'], self.node[o]['x']), (self.node[d]['y'], self.node[d]['x']), unit=Unit.METERS)


    def a_satr(self, origin, dest, max_layer=500, verbose=False, plot=False):
        """Route planning by A star algm

        Args:
            origin ([type]): [description]
            dest ([type]): [description]
            verbose (bool, optional): [description]. Defaults to False.
            plot (bool, optional): [description]. Defaults to False.

        Returns:
            dict: The route planning result with path, cost and status.
            status_dict = {-1: 'unreachable'}
        """
        if (origin, dest) in self.route_planning_memo:
            res = self.route_planning_memo[(origin, dest)]
            return res
        
        if origin not in self.graph or dest not in self.graph:
            print(f"Edge({origin}, {dest})\
                {', origin not in graph' if origin not in self.graph else ', '}\
                {', dest not in graph' if dest not in self.graph else ''}")
            return None

        frontier = [(0, origin)]
        came_from, distance = {}, {}
        came_from[origin] = None
        distance[origin] = 0

        # TODO: add `visited`
        layer = 0
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == dest or layer > max_layer:
                break
            
            for nxt in self.graph[cur]:
                if nxt not in self.graph:
                    continue
                
                new_cost = distance[cur] + self.edge[(cur, nxt)]
                if nxt not in distance or new_cost < distance[nxt]:
                    distance[nxt] = new_cost
                    if distance[nxt] > 10**4:
                        continue

                    heapq.heappush(frontier, (new_cost+self.cal_nodes_dis(dest, nxt), nxt) )
                    came_from[nxt] = cur
            layer += 1

        if cur != dest:
            res = {'path': None, 'cost': np.inf, "status": -1} 
            self.route_planning_memo[(origin, dest)] = res
            return res

        # reconstruct the route
        route, queue = [dest], deque([dest])
        while queue:
            node = queue.popleft()
            # assert node in came_from, f"({origin}, {dest}), way to {node}"
            if came_from[node] is None:
                continue
            route.append(came_from[node])
            queue.append(came_from[node])
        route = route[::-1]

        res = {'path':route, 'cost': distance[dest], 'status':1}
        self.route_planning_memo[(origin, dest)] = res

        if plot:
            path_lst = gpd.GeoDataFrame([ { 's': route[i], 'e': route[i+1]} for i in range(len(route)-1) ])
            ax = path_lst.merge(self.df_edges, on=['s', 'e']).plot()
                    
        return res


    @property
    def df_node_with_degree(self,):
        return self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()


def load_graph_helper(bbox=None, xml_fn=None, combine_link=True, overwrite=False, cache_folder='../cache', convert_to_geojson=False):
    """ parse xml to edge and node with/without combiantion"""
    if xml_fn is not None:
        net = Digraph_OSM(xml_fn=xml_fn, combine_link=combine_link)
        return net
    
    """ Read Digraph_OSM object from file """
    assert isinstance(bbox, list), 'Check input bbox'
    
    bbox_str = '_'.join(map(str, bbox))
    fn = os.path.join(cache_folder, f"net_{bbox_str}.pkl")
    s = PickleSaver()
    
    if os.path.exists(fn) and not overwrite:
        net = s.read(fn)
    else:
        net = Digraph_OSM(bbox=bbox, combine_link=combine_link)
        s.save(net, fn)
        if convert_to_geojson:
            gdf_to_geojson(net.df_edges, f'../cache/edges_{bbox_str}')
            gdf_to_geojson(net.df_nodes, f'../cache/nodes_{bbox_str}')
    
    return net


net = load_graph_helper(xml_fn='../input/futian.xml', combine_link=True)
# net = load_graph_helper(bbox=SZ_BBOX, combine_link=True, convert_to_geojson=False)

#%%

""" functions """
def load_trajectory(fn = '../input/tra.shp'):
    tra = gpd.read_file(fn, encoding='utf-8')
    tra.set_crs('EPSG:4326', inplace=True)
    if 'time' in tra.columns:
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
    df_candidates = boxes.apply(lambda x: edges.sindex.query(x, predicate='intersects')).explode().dropna()
    
    if df_candidates.shape[0] == 0:
        return None
    
    df_candidates = pd.DataFrame(df_candidates).reset_index().rename(columns={'index': 'pid', 'geometry':'rindex'})
    df_candidates = df_candidates.merge(traj['geometry'], left_on='pid', right_index=True)\
                              .merge(edges['geometry'], left_on='rindex', right_index=True)\
                              .rename(columns={'geometry_x': 'point_geom', 'geometry_y': 'edge_geom'})

    # set_crs: out_sys=32649;  CRS(f"EPSG:{out_sys}")
    df_candidates.loc[:, 'dist_to_line'] = df_candidates.apply(lambda x: x.point_geom.distance(x.edge_geom) / DIS_FACTOR, axis=1)


    if filter:
        candidates_filtered = _filter_candidate(df_candidates, top_k)
        
    if plot:
        ax = edges.loc[df_candidates.rindex.values].plot()
        if filter:
            edges.loc[candidates_filtered.rindex.values].plot(ax=ax, color='red')
        traj.plot(ax=ax)

    return candidates_filtered if filter else df_candidates.sort_values(['pid', 'dist_to_line']).groupby('pid').head(5)


def cal_observ_prob(df, std_deviation=10):
    observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)
    
    def helper(x):
        return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))
    
    df.loc[:, 'observ_prob'] = df.dist_to_line.apply( helper)
    df.loc[:, 'observ_prob'] = df.observ_prob / df.observ_prob.max()

    return df


def cal_relative_offset(node:Point, polyline:LineString, verbose=False):
    """Calculate the relative offset between the node's foot and the polyline.

    Args:
        node (Point): [description]
        polyline (LineString): [description]
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
        
    Example: 
        ```
        from shapely import wkt
        node = wkt.loads('POINT (114.051228 22.539597)')
        linestring = wkt.loads('LINESTRING (114.0516508 22.539516, 114.0515715 22.5395317, 114.0515222 22.5395533, 114.0514758 22.5395805, 114.0514441 22.5396039, 114.0514168 22.5396293, 114.0513907 22.5396616, 114.0513659 22.5396906, 114.0513446 22.5397236, 114.0512978 22.5398214)')
        cal_relative_offset(node, linestring)
        ```
    """
    lines = [LineString((polyline.coords[i], polyline.coords[i+1])) for i in range(len(polyline.coords)-1) ]
    lines = gpd.GeoDataFrame( {'geometry': lines} )

    lines.loc[:, 'dist'] = lines.geometry.apply(lambda x: coords_pair_dist(x.coords[0], x.coords[-1], xy=True))
    lines.loc[:, 'foot_info'] = lines.apply(lambda x: cal_foot_point_on_polyline(node, x.geometry), axis=1)
    lines.loc[:, 'foot_factor'] = lines.loc[:, 'foot_info'].apply(lambda x: x['flag'])

    dist_prev_line, offset = 0, 0
    for i, l in lines.iterrows():
        if l.foot_factor > 1:
            dist_prev_line += l.dist
            continue
        _dist = coords_pair_dist(l.foot_info['foot'], l.geometry.coords[0], xy=True)
        
        if l.foot_factor > 0:
            offset = dist_prev_line + _dist
            break
        
        if l.foot_factor <= 0:
            offset = dist_prev_line - _dist
            break

    return offset


def linestring_combine_helper(path, net):
    """Create Linestring by coords id sequence.

    Args:
        path (list): The id sequence of Coordinations.
        net (Digraph_OSM): The Digraph_OSM object.

    Returns:
        Linestring: The linstring of the speicla sequence.
    """
    if path is None or len(path) <= 1:
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
    for _, sub in df_candidates.groupby('pid'):
        tList.append(sub)

    for i in range(len(tList)-1):
        base_atts = ['pid', 'rindex','s', 'e', 'offset']
        a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
        a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1
        graph_t.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )

    graph_t = pd.concat(graph_t).reset_index(drop=True)
    graph_t.loc[:, 'shortest_path'] = graph_t.apply(lambda x: net.a_satr(x.e_0, x.s_1, plot=False), axis=1)
    graph_t.loc[:, 'd_sht']   = graph_t.shortest_path.apply(lambda x: x['cost'] if x is not None else np.inf )
    graph_t.loc[:, 'd_euc']   = graph_t.apply(lambda x: coords_pair_dist(traj.loc[x.pid_0].geometry, traj.loc[x.pid_1].geometry), axis=1)
    graph_t.loc[:, 'd_step0'] = graph_t.apply(lambda x: net.cal_nodes_dis(x.s_0, x.e_0), axis=1)
    graph_t.loc[:, 'w']       = graph_t.d_sht + (graph_t.d_step0 - graph_t.offset_0) + graph_t.offset_1 
    
    graph_t.loc[:, 'v'] = graph_t.apply(lambda x:  x.d_euc/x.w if x.d_euc/x.w <= 1 else 1, axis=1 )  
    
    # The case: o and d all on the same link
    # ValueError: Must have equal len keys and value when setting with an iterable, http://www.cocoachina.com/articles/81838
    graph_t.loc[graph_t.rindex_0 == graph_t.rindex_1, ['v', 'shortest_path']] = [1, None]

    # create geometry
    graph_t.loc[:, 'geometry'] = graph_t.shortest_path.apply(lambda x: linestring_combine_helper(x['path'], net) if x is not None else None)
    graph_t = gpd.GeoDataFrame(graph_t)

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
                # assert graph_t.query(f'rindex_0=={prev.rindex} and rindex_1 == {cur.rindex}').shape[0] == 1, f'rindex_0=={prev.rindex} and rindex_1 == {cur.rindex}'
                v = graph_t.query(f'rindex_0=={prev.rindex} and rindex_1 == {cur.rindex}').iloc[0].v
                alt = f_score[k] + v
                # print(i, j, v)
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
    # TODO: The nodes along the path in sequence.
    steps = gpd.GeoDataFrame(rList.merge(net.df_edges, on=['s','e']))

    rList.loc[:, 'nxt_s'] = rList.s.shift(-1).fillna(0).astype(np.int)
    links = rList.merge( graph_t, left_on=['e', 'nxt_s'], right_on=['e_0', 's_1'] )[['s','e', 'nxt_s', 'geometry']]

    steps.reset_index(inplace=True)
    links.reset_index(inplace=True)
    links.loc[:,'index'] = .5 + links.loc[:,'index']

    path = steps.append(links).sort_values('index')

    if plot: path.plot()
    
    return path[['index', 's', 'e', 'name', 'rid', 'lanes', 'geometry']]


def st_matching(traj, net, plot=True, debug=False):
    # step 1: candidate prepararation
    if traj.shape[0] == 0:
        return None
    if traj.shape[0] == 1:
        #TODO The cloesest edge.
        return 
    
    df_candidates = get_candidates(traj, net.df_edges, georadius=50, plot=True, verbose=False, filter=True)
    if df_candidates is None:
        return None

    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates)

    # step 2.2: Spatial analysis, transmission prob
    tList, graph_t = cal_trans_prob(df_candidates, net)

    # TODO step 3: temporal analysis

    # step 4: find matched sequence
    rList = find_matched_sequence(graph_t, df_candidates, tList)
    path = get_whole_path(rList, graph_t, net, False)

    if debug:
        matching_debug(tList, graph_t, net, debug)

    if plot:
        _, ax = map_visualize(traj, alpha=.5)
        # traj.plot(ax=ax, marker = '*', color='red', zorder=9, label= 'Point')
        # edge_lst = net.df_edges.sindex.query(box(*traj.total_bounds), predicate='intersects')
        # net.df_edges.loc[edge_lst].plot( color='black',linestyle='--', linewidth=.8, alpha=.8, label='Network' )
        
        net.df_edges.loc[df_candidates.rindex.values].plot(
            ax=ax, label='Candidate', color='blue', linestyle='--', linewidth=.8,alpha=.8)
        path.plot(ax=ax, label='Path', color='red', alpha=.8)
        plt.legend()
    
    return path


"""" matching plot debug helper """
def matching_debug_subplot(item, net, ax=None, legend=True):
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
    gpd.GeoDataFrame( item ).T.plot(ax=ax, color='red', label='path')
    net.df_edges.loc[[j]].plot(ax=ax, linestyle='--', label='last step', color='black')

    # aux
    ax.set_title( f"{item.rindex_0} -> {item.rindex_1}, V: {item.v:.3f}", color = 'black' if item.v < 0.7 else 'red' )
    ax.set_axis_off()
    if legend: ax.legend()
    
    return ax
    

def matching_debug_level(tList, graph_t, level, net, debug=False, debug_folder=DEBUG_FOLDER):
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
    
    if debug:
        t = time.strftime("%Y%m%d_%H", time.localtime()) 
        plt.savefig( os.path.join(debug_folder, f"{t}_level_{level}.jpg"), dpi=300)
        plt.close()
        
    return True


def matching_debug(tList, graph_t, net, debug=True):
    levels = len(tList) - 1
    for level in range(levels):
        matching_debug_level(tList, graph_t, level, net=net, debug=debug)
    
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
"""
# TODO 
    sampleing points; 
    trim the first point and the last point; 
    sequence coordination of path
"""

if True:
    # st_matching(traj, net, False)
    # error_lst = [ 37, 38, 40]
    # planning_error = [43, 50, 56] 
    # key_error = [64]
    # uncontinous = [ 31, 40, 42,69, 76, 119] # 修改filter后解决问题

    idx = 37
    traj = split_line_to_points(df_path.iloc[idx].geometry)
    traj = traj[traj.index%1==0].reset_index(drop=True)

    # step 1: candidate prepararation
    df_candidates = get_candidates(traj, net.df_edges, georadius=50, plot=True, verbose=False, filter=True)

    # step 2.1: Spatial analysis, obervation prob
    cal_observ_prob(df_candidates)

    # step 2.2: Spatial analysis, transmission prob
    tList, graph_t = cal_trans_prob(df_candidates, net)


    # step 4: find matched sequence
    rList = find_matched_sequence(graph_t, df_candidates, tList)
    path = get_whole_path(rList, graph_t, net, True)

    graph_t


#%%
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
    traj = load_trajectory("../input/traj_0.geojson")
    st_matching(traj, net, plot=True, debug=False)

