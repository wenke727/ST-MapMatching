#%%
import os
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from geo.geo_helper import edge_parallel_offset
from geo.osm_helper import download_osm_xml, parse_xml_to_topo, combine_links_parallel_helper

from utils.pickle_helper import Saver
from utils.logger_helper import make_logger
from utils.parallel_helper import parallel_process
from graph.geograph import GeoDigraph

from db.db_process import gdf_to_postgis

from setting import filters as way_filters
from setting import SZ_BBOX, GBA_BBOX, PCL_BBOX, FT_BBOX, LOG_FOLDER, CACHE_FOLDER

warnings.filterwarnings('ignore')

#%%

class DigraphOSM(GeoDigraph, Saver):
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

        if bbox is not None:
            os.makedirs(CACHE_FOLDER, exist_ok=True)
            xml_fn = f"{CACHE_FOLDER}/osm_{'_'.join(map(str, bbox))}.xml"
        assert download_osm_xml(xml_fn, bbox, True), "check `download_osm_xml`"

        # topo data
        # TODO gcj, wgs
        self.df_nodes, self.df_edges = parse_xml_to_topo(xml_fn, road_info_fn, type_filter=self.road_type_filter, crs=self.crs_wgs)
        if "traffic_signals" not in self.df_nodes.columns:
            self.df_nodes.loc[:, 'traffic_signals'] = np.nan
        self.signal_control_points = self.df_nodes[~self.df_nodes.traffic_signals.isna()].index.unique()
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
        return DeprecationWarning


    """ aux func """
    def _add_reverse_edge(self, df_edges):
        return DeprecationWarning
        

    def _combine_rids(self):   
        return DeprecationWarning


    def _edge_offset(self,):
        return DeprecationWarning


    """ API """
    def merge_edge(self, df, on=['s', 'e']):
        return self.df_edges.merge(df, on=on)



#%%
if __name__ == '__main__':
    # create new network
    # net = DigraphOSM("Shenzhen", bbox=SZ_BBOX)
    # net.save()

    # Resume from pkl
    net = DigraphOSM("Shenzhen", resume='../input/ShenzhenNetwork.pkl')

    # route planning  
    path = net.route_planning(o=7959990710, d=499265789)
    path['gdf'].plot()

# %%
