#%%
import os
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from haversine import haversine, Unit
from shapely.geometry import LineString

from geo.geo_helper import edge_parallel_offset
from geo.osm_helper import download_osm_xml, parse_xml_to_topo, combine_links_parallel_helper

from utils.pickle_helper import Saver
from utils.logger_helper import make_logger
from utils.parallel_helper import parallel_process
from utils.DataStructure.digraph import DigraphAstar

from db.db_process import gdf_to_postgis

from setting import filters as way_filters
from setting import SZ_BBOX, GBA_BBOX, PCL_BBOX, FT_BBOX, LOG_FOLDER, CACHE_FOLDER

warnings.filterwarnings('ignore')

#%%

class DigraphOSM(DigraphAstar, Saver):
    def __init__(self, 
                 name, 
                 resume=None, 
                 bbox=None, 
                 xml_fn=None, 
                 combine=True, 
                 reverse_edge=True,
                 two_way_offset=True,
                 road_info_fn='../input/osm_road_speed.csv',
                 n_jobs=8, 
                 *args, **kwargs):
        if resume is not None:
            if self.resume(resume): 
                return
        
        assert not(bbox is None and xml_fn is None), "Define one of the bbox or the xml path."

        # config
        self.name     = name
        self.n_jobs   = n_jobs
        self.crs_wgs  = 4326
        self.crs_prj  = 900913
        self.logger   = make_logger(LOG_FOLDER, "INFO")
        self.df_nodes = None
        self.df_edges = None
        
        self.road_type_filter   = way_filters['auto']['highway']
        self.node_pair_dis_memo = {} # self.cal_nodes_dis()

        if bbox is not None:
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            xml_fn = f"{CACHE_FOLDER}/osm_{'_'.join(map(str, bbox))}.xml"
        assert download_osm_xml(xml_fn, bbox, True), "check `download_osm_xml`"

        # topo data
        # TODO gcj, wgs
        self.df_nodes, self.df_edges = parse_xml_to_topo(xml_fn, road_info_fn, type_filter=self.road_type_filter, crs=self.crs_wgs)
        if "traffic_signals" not in self.df_nodes.columns:
            self.df_nodes.loc[:, 'traffic_signals'] = np.nan
        self.traffic_signals = self.df_nodes[~self.df_nodes.traffic_signals.isna()].index.unique()
        DigraphAstar.__init__(self, self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)
        Saver.__init__(self, f"{CACHE_FOLDER}/{name}.pkl")

        if combine:
            self.df_edges = self._combine_rids()
        if reverse_edge:
            self.df_edges = self._add_reverse_edge(self.df_edges)
        self.df_edges.loc[:, 'eid'] = self.df_edges.index
        
        if combine or reverse_edge:
            # Warning: The nodes are compressed to speed up
            DigraphAstar.__init__(self, self.df_edges[['s', 'e', 'dist']].values, self.df_nodes.to_dict(orient='index'), *args, **kwargs)
        
        if reverse_edge and two_way_offset:
            self.df_edges = self._edge_offset()
        
        order_atts = ['eid', 'rid', 'name', 'order', 's', 'e', 'waypoints', 'road_type', 'dir', 'lanes', 'dist', 'oneway', 'is_ring', 'geometry', 'geom_origin']
        self.df_edges = self.df_edges[[i for i in  order_atts if i in self.df_edges.columns]]
        self.od_to_coords = self.df_edges[['s', 'e', 'geom_origin']].set_index(['s', 'e']).geom_origin.apply(lambda x: x.coords[:]).to_dict()
        
        return


    def resume(self, fn):
        """ resume Digraph_OSM from the file """
        assert os.path.exists(fn), "Double check the file"

        Saver.__init__(self, fn)
        self._load(fn)
        self.logger = make_logger(LOG_FOLDER, "INFO")
        
        if type(self.df_edges.iloc[0].geom_origin) == str:
            self.df_edges.geom_origin = self.df_edges.geom_origin.apply(wkt.loads)
        if not hasattr(self, "od_to_coords"):
            self.od_to_coords = self.df_edges[['s', 'e', 'geom_origin']].set_index(['s', 'e']).geom_origin.apply(lambda x: x.coords[:]).to_dict()
        
        print(f"load suceess, the pkl was created at {self.create_time}")

        return True


    def save(self):
        self._save()


    def route_planning(self, o, d, *args, **kwargs):
        route = self.a_star(o, d, *args, **kwargs)
        route['gdf'] = self.node_seq_to_df_edge(route['path'])
       
        return route


    def cal_nodes_dis(self, o, d):
        assert o in self.node and d in self.node, "Check the input o and d."
        if (o, d) in self.node_pair_dis_memo:
            return self.node_pair_dis_memo[(o, d)]
        
        return haversine((self.node[o]['y'], self.node[o]['x']), (self.node[d]['y'], self.node[d]['x']), unit=Unit.METERS)


    def to_postgis(self, name):
        try:
            gdf_to_postgis(self.df_edges, f'topo_osm_{name}_edge')
            gdf_to_postgis(self.df_node_with_degree, f'topo_osm_{name}_endpoint')
            
            self.df_nodes.loc[:, 'nid'] = self.df_nodes.index
            self.df_nodes = self.df_nodes[['nid', 'x', 'y', 'traffic_signals', 'geometry']]
            gdf_to_postgis(self.df_nodes, f'topo_osm_{name}_node')
            return True
        except:
            if self.logger is not None:
                self.logger.error('upload data error.')
        
        return False


    def to_csv(self, name, folder = None):
        try:
            edge_fn = f'topo_osm_{name}_edge.csv'
            node_fn = f'topo_osm_{name}_endpoint.csv'
            
            if folder is not None:
                edge_fn = os.path.join(edge_fn, edge_fn)
                node_fn = os.path.join(edge_fn, node_fn)
            
            df_edges = self.df_edges.copy()
            atts = ['eid', 'rid', 'name', 's', 'e', 'order', 'road_type', 'dir', 'lanes', 'dist', 'oneway', 'geom_origin']
            pd.DataFrame(df_edges[atts].rename({'geom_origin': 'geom'})).to_csv(edge_fn, index=False)
            
            df_nodes = self.df_nodes.copy()
            df_nodes.loc[:, 'nid'] = df_nodes.index
            df_nodes.loc[:, 'geom'] = df_nodes.geometry.apply(lambda x: x.to_wkt())
            
            atts = ['nid', 'x', 'y', 'traffic_signals', 'geometry']
            pd.DataFrame(df_nodes[atts].rename({'geom_origin': 'geom'})).to_csv(node_fn, index=False)
            
            return True
        except:
            if self.logger is not None:
                self.logger.error('upload data error.')
        
        return False


    @property
    def df_node_with_degree(self,):
        # Returns the point layer of topology data.
        return self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()


    """ aux func """
    def _add_reverse_edge(self, df_edges):
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
        df_edges.loc[:, 'dir'] = 1

        df_edge_rev = df_edges.query('oneway == False and not is_ring')
        df_edge_rev.loc[:, 'dir']       = -1
        df_edge_rev.loc[:, 'order']     = -df_edge_rev.order - 1
        df_edge_rev.loc[:, 'geometry']  =  df_edge_rev.geometry.apply(lambda x: LineString(x.coords[::-1]) )
        df_edge_rev.loc[:, 'waypoints'] =  df_edge_rev.waypoints.apply(lambda x: ",".join(x.split(",")[::-1]))
        df_edge_rev.rename(columns={'s':'e', 'e':'s'}, inplace=True)

        return df_edges.append(df_edge_rev).reset_index(drop=True)


    def _get_not_node_points(self):
        """Identify the road segment with nodes of 1 indegree and 1 outdegree. 

        Returns:
            [list]: Road segement list.
        """
        return self.degree.query( "indegree == 1 and outdegree == 1" ).index.unique().tolist()
    

    def _combine_rids(self):
        omit_pids    = [x for x in self._get_not_node_points() if x not in self.traffic_signals]
        omit_records = self.df_edges.query( f"s in @omit_pids" )
        omit_rids    = omit_records.rid.unique().tolist()
        keep_records = self.df_edges.query( f"rid not in @omit_rids" )

        omit_pid_dict = omit_records.sort_values(by=["rid", 'order']).groupby('rid').order.apply(list).to_dict()

        """ Data Parallel """
        df = self.df_edges.query("rid in @omit_rids").sort_values(by=['rid', 'order'])
        df.loc[:, 'part'] = df.rid % self.n_jobs
        df_parts = df.groupby('part')

        res = parallel_process(combine_links_parallel_helper, zip(df_parts, [self.df_nodes]*len(df_parts), [omit_pid_dict]*len(df_parts)), n_jobs=self.n_jobs)
        comb_rids = gpd.GeoDataFrame(pd.concat(res), crs=f"EPSG:{self.crs_wgs}")

        keep_records.loc[:, 'waypoints'] = keep_records.apply(lambda x: f"{x.s},{x.e}",axis=1)
        comb_rids = comb_rids.append(keep_records, sort=['rid', 'order'], ignore_index=True)[list(keep_records)]
        
        # origin function 
        # res = []
        # for rid in tqdm(omit_rids, 'Combine links \t'):
        #     res.append(self.combine_links_of_rid(rid, omit_records, self.df_edges))

        # comb_rids = gpd.GeoDataFrame(pd.concat(res), crs=f"EPSG:{self.crs_wgs}")
        # comb_rids = keep_records.append(comb_rids).reset_index(drop=True)

        return comb_rids


    def _edge_offset(self,):
        df_edge = self.df_edges.copy()
        
        df_edge.loc[:, 'geom_origin'] = df_edge.geometry.copy()
        # df_edge.loc[:, 'geom_origin'] = df_edge.geometry.apply(lambda x: x.to_wkt())
        geom_offset = df_edge[~df_edge.oneway].apply( lambda x: edge_parallel_offset(x, logger=self.logger), axis=1 )
        df_edge.loc[geom_offset.index, 'geometry'] = geom_offset

        return df_edge


    """ API """
    def node_seq_to_df_edge(self, node_lst, on=['s', 'e'], attrs=None):
        """Convert the id sequence of nodes into an ordered edges. 

        Args:
            node_lst (list): The id sequence of nodes
            on (list, optional): The column names of `origin` and `destination` to join on.. Defaults to ['s', 'e'].

        Returns:
            [type]: [description]
        """
        if len(node_lst) <= 1:
            return None
        
        df = gpd.GeoDataFrame([ {on[0]: node_lst[i], on[1]: node_lst[i+1]} for i in range(len(node_lst)-1) ])
        
        if attrs is None:
            return df.merge(self.df_edges, on=on)
        
        return df.merge(self.df_edges, on=on)[attrs]


    def node_seq_to_polyline(self, node_lst:list):
        """Create Linestring by node id sequence.

        Args:
            path (list): The id sequence of Coordinations.
            net (Digraph_OSM): The Digraph_OSM object.

        Returns:
            Linestring: The linstring of the speicla sequence.
        """
        if node_lst is None or len(node_lst) <= 1 or np.isnan(node_lst).all():
            return None
        
        points = np.array([self.od_to_coords[(node_lst[i], node_lst[i+1])]for i in range(len(node_lst)-1)])
        res = list(points[0])
        for ps in points[1:]:
            res += list(ps[1:])
        
        return LineString(res)       
        

    def get_edge(self, eid, att=None):
        """Get edge by eid [0, n]"""
        res = self._get_feature('df_edges', eid, att)
        if att == 'waypoints':
            res = [int(i) for i in res.split(",")]
            
            return res
        
        return res


    def get_node(self, nid, att=None):
        return self._get_feature('df_nodes', nid, att)
    

    def _get_feature(self, df_name, id, att=None):
        """get edge by id.

        Args:
            id (_type_): _description_
            att (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        df = getattr(self, df_name)
        if df is None:
            print(f"Dosn't have the attibute {df_name}")
            return None

        if isinstance(id, int) and id not in df.index:
            return None

        if isinstance(id, list) or isinstance(id, tuple) or isinstance(id, np.ndarray):
            for i in id:
                if i not in df.index:
                    return None
        
        res = df.loc[id]
        return res if att is None else res[att]
        

    def merge_edge(self, df, on=['s', 'e']):
        return self.df_edges.merge(df, on=on)


    def spatial_query(self, geofence, name='df_edges', predicate='intersects'):
        gdf = getattr(self, name)
        
        return gdf.sindex.query(geofence, predicate=predicate)
        

#%%
if __name__ == '__main__':
    # create new network
    # net = DigraphOSM("GBA", bbox=GBA_BBOX)
    # net = DigraphOSM("PCL", bbox=PCL_BBOX)
    # net = DigraphOSM("Shenzhen", bbox=SZ_BBOX)
    # net = DigraphOSM("Futian", bbox=FT_BBOX)
    # net.save()

    # Resume from pkl
    # net = DigraphOSM("Shenzhen", resume='../cache/SZ.pkl')
    net = DigraphOSM("Shenzhen", resume='E:\code\ST-MapMatching\cache\SZ.pkl')

    # route planning  
    path = net.route_planning(o=7959990710, d=499265789)

    from tilemap import plot_geodata
    plot_geodata(path['gdf'])
    
    # save data to db
    # net.to_postgis('shenzhen')


# %%
