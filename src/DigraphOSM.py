#%%
import os
import xml
import heapq
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely import wkt
import geopandas as gpd
from xml.dom import minidom
from collections import deque
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from shapely.geometry import Point, LineString, box

from utils.classes import Digraph
from utils.pickle_helper import PickleSaver
from utils.log_helper import LogHelper, logbook
from utils.interval_helper import merge_intervals
from coords.coordTransfrom_shp import coord_transfer
from utils.geo_helper import gdf_to_geojson, gdf_to_postgis, edge_parallel_offset

from setting import filters as way_filters
from setting import SZ_BBOX, GBA_BBOX, PCL_BBOX

warnings.filterwarnings('ignore')

# TODO net.df_edges: index -> eid

#%%
class Digraph_OSM(Digraph):
    def __init__(self, 
                 bbox=None,
                 xml_fn='../input/futian.xml', 
                 road_info_fn='../input/osm_road_speed.xlsx', 
                 combine_link=True,
                 reverse_edge=True,
                 two_way_offeset=True,
                 logger=None,
                 upload_to_db='shenzhen',
                 *args, **kwargs):
        assert not(bbox is None and xml_fn is None), "Please define one of the bbox or the xml path."
        if bbox is not None:
            xml_fn = f"../cache/osm_{'_'.join(map(str, bbox))}.xml"
            self.download_map(xml_fn, bbox, True)
        
        self.df_nodes, self.df_edges = self.get_road_network(xml_fn, road_info_fn)
        self.node_dis_memo = {}
        self.route_planning_memo = {}
        self.logger = logger
        super().__init__(self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)

        self.df_edges.set_crs('EPSG:4326', inplace=True)
        self.df_nodes.set_crs('EPSG:4326', inplace=True)
                
        if combine_link:
            self.df_edges = self.combine_rids()
            self.df_edges.reset_index(drop=True, inplace=True)

        if reverse_edge:
            self.df_edges = self.add_reverse_edge(self.df_edges)
            self.df_edges.reset_index(drop=True, inplace=True)
        
        self.df_edges.loc[:, 'eid'] = self.df_edges.index
        if combine_link or reverse_edge:
            # self.df_nodes = self.df_nodes.loc[ np.unique(np.hstack((self.df_edges.s.values, self.df_edges.e.values))),:]
            super().__init__(self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)
        
        if two_way_offeset:
            self.df_edges = self.edge_offset()
        
        order_atts = ['eid', 'rid', 'name', 's', 'e', 'order', 'road_type', 'dir', 'lanes', 'dist', 'oneway', 'is_ring', 'geometry', 'geom_origin']
        self.df_edges = self.df_edges[order_atts]


    def download_map(self, fn, bbox, verbose=False):
        """Download OSM map of bbox from Internet.

        Args:
            fn (function): [description]
            bbox ([type]): [description]
            verbose (bool, optional): [description]. Defaults to False.
        """
        if os.path.exists(fn):
            return

        if verbose:
            print("Downloading {}".format(fn))
        
        if isinstance(bbox, list) or isinstance(bbox, np.array):
            bbox = ",".join(map(str, bbox))

        import requests
        url = f'http://overpass-api.de/api/map?bbox={bbox}'
        r = requests.get(url, stream=True)
        with open(fn, 'wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)

        if verbose:
            print("Downloaded success.\n")

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
            ndlist = way.getElementsByTagName('nd')
            nds = []
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
        edges.sort_values(['rid', 'order'], inplace=True)

        # nodes filter
        ls = np.unique(np.hstack((edges.s.values, edges.e.values)))
        nodes = nodes.loc[ls,:]

        if fn_road:
            road_speed = pd.read_excel(fn_road)[['road_type', 'v']]
            edges = edges.merge( road_speed, on ='road_type' )
        
        keep_cols = [i for i in keep_cols if i in edges.columns]
        
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
            # reversible, alternating: https://wiki.openstreetmap.org/wiki/Tag:oneway%3Dreversible
            if oneway_flag == 'yes' or oneway_flag == '1' or oneway_flag == True:
                flag = True
            elif oneway_flag == '-1':
                flag = True
                # way.is_reversed = True
            elif oneway_flag == 'no' or oneway_flag == '0' or oneway_flag == False:
                flag = False
            elif oneway_flag in ['reversible', 'alternating']:
                flag = False
            else:
                flag = False
                if self.logger is not None:
                    self.logger.warning(f'new road type detected at: {oneway_flag}')

            return flag

        df_edges.oneway = df_edges.oneway.fillna('no').apply(_juedge_oneway)
        df_edges.loc[:, 'is_ring'] = df_edges.geometry.apply( lambda x: x.is_ring)

        df_edge_rev = df_edges.query('oneway == False and not is_ring')
        df_edge_rev.loc[:, 'order']    = -df_edge_rev.order - 1
        df_edge_rev.loc[:, 'geometry'] =  df_edge_rev.geometry.apply( lambda x: LineString(x.coords[::-1]) )
        df_edge_rev.rename(columns={'s':'e', 'e':'s'}, inplace=True)

        df_edge_rev.loc[:, 'dir'] = -1
        df_edges.loc[:, 'dir'] = 1

        return df_edges.append(df_edge_rev).reset_index(drop=True)


    def get_intermediate_point(self):
        """Identify the road segment with nodes of 1 indegree and 1 outdegree. 

        Returns:
            [list]: Road segement list.
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
        comb_rids = keep_records.append(comb_rids).reset_index(drop=True)

        return comb_rids


    def edge_offset(self,):
        df_edge = self.df_edges.copy()
        
        df_edge.loc[:, 'geom_origin'] = df_edge.geometry.apply(lambda x: x.to_wkt())
        geom_offset = df_edge[~df_edge.oneway].apply( lambda x: edge_parallel_offset(x, logger=self.logger), axis=1 )
        df_edge.loc[geom_offset.index, 'geometry'] = geom_offset

        return df_edge


    def cal_nodes_dis(self, o, d):
        assert o in self.node and d in self.node, "Check the input o and d."
        if (o, d) in self.node_dis_memo:
            return self.node_dis_memo[(o, d)]
        
        return haversine((self.node[o]['y'], self.node[o]['x']), (self.node[d]['y'], self.node[d]['x']), unit=Unit.METERS)


    def a_star(self, origin, dest, max_layer=500, max_dist=10**4, verbose=False, plot=False):
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
            print(f"Edge({origin}, {dest})",
                f"{', origin not in graph' if origin not in self.graph else ', '}",
                f"{', dest not in graph' if dest not in self.graph else ''}")
            return None

        frontier = [(0, origin)]
        came_from, distance = {}, {}
        came_from[origin] = None
        distance[origin] = 0

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
                    if distance[nxt] > max_dist:
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
        # Returns the point layer of topology data.
        return self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()


    def upload_topo_data_to_db(self, name):
        try:
            gdf_to_postgis(self.df_edges, f'topo_osm_{name}_edge')
            gdf_to_postgis(self.df_node_with_degree, f'topo_osm_{name}_endpoint')
            
            self.df_nodes.loc[:, 'nid'] = self.df_nodes.index
            self.df_nodes = self.df_nodes[['nid', 'x', 'y', 'traffic_signals', 'geometry']]
            gdf_to_postgis(self.df_nodes, f'topo_osm_{name}_node')
            return True
        except:
            if self.logger is not None:
                logger.error('upload data error.')
        
        return False


    def node_sequence_to_edge(self, node_lst, on=['s', 'e'], attrs=None):
        """Convert the id sequence of nodes into an ordered edges. 

        Args:
            node_lst (list): The id sequence of nodes
            on (list, optional): The column names of `origin` and `destination` to join on.. Defaults to ['s', 'e'].

        Returns:
            [type]: [description]
        """
        df = gpd.GeoDataFrame([ {on[0]: node_lst[i], on[1]: node_lst[i+1]} for i in range(len(node_lst)-1) ])
        
        if attrs is None:
            return df.merge(self.df_edges, on=on)
        
        return df.merge(self.df_edges, on=on)[attrs]


def load_net_helper(bbox=None, xml_fn=None, combine_link=True, overwrite=False, reverse_edge=True, cache_folder='../cache', convert_to_geojson=False, logger=None, two_way_offeset=True):
    # parse xml to edge and node with/without combiantion
    if xml_fn is not None:
        net = Digraph_OSM(xml_fn=xml_fn, combine_link=combine_link)
        return net
    
    # Read Digraph_OSM object from file
    assert isinstance(bbox, list), 'Check input bbox'
    
    bbox_str = '_'.join(map(str, bbox))
    fn = os.path.join(cache_folder, f"net_{bbox_str}.pkl")
    s = PickleSaver()
    
    if os.path.exists(fn) and not overwrite:
        net = s.read(fn)
        net.df_edges.geom_origin = net.df_edges.geom_origin.apply(wkt.loads)
    else:
        net = Digraph_OSM(bbox=bbox, combine_link=combine_link, reverse_edge=reverse_edge, two_way_offeset=two_way_offeset, logger=logger)
        s.save(net, fn)
        if convert_to_geojson:
            gdf_to_geojson(net.df_edges, f'../cache/edges_{bbox_str}')
            gdf_to_geojson(net.df_nodes, f'../cache/nodes_{bbox_str}')
    
    return net


#%%
if __name__ == '__main__':
    logger = LogHelper(log_name='digraph_osm.log', log_dir='../log', stdOutFlag=False).make_logger(level=logbook.INFO)
    net = load_net_helper(
        bbox=SZ_BBOX, 
        overwrite=True, 
        logger=logger,
        combine_link=True, 
        reverse_edge=True, 
        two_way_offeset=True, 
    )
    net.upload_topo_data_to_db('shenzhen')

    
    """ a_star algorithm test """
    path = net.a_star(1491845212, 1116467141, max_layer=10**5, max_dist=20**7)

    """ construct trajectories """ 
    # gdf_path = net.node_sequence_to_edge(path['path'])

# %%
