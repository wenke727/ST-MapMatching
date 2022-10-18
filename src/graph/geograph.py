import os
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

import sys
sys.path.append('../')
sys.path.append('./src')
from graph.base import Digraph
from graph.bfs import a_star
from osmnet.osm_io import load_graph
from utils.serialization import save_checkpoint, load_checkpoint

#%%
class GeoDigraph(Digraph):
    def __init__(self, df_edges:GeoDataFrame=None, df_nodes:GeoDataFrame=None, *args, **kwargs):
        self.df_edges = df_edges
        self.df_nodes = df_nodes
        self.search_memo = {}
        self.nodes_dist_memo = {}
        
        if df_edges is not None and df_nodes is not None:
            super().__init__(df_edges[['src', 'dst', 'dist']].values, df_nodes.to_dict(orient='index'), *args, **kwargs)
        

    def search(self, src, dst, max_steps=2000, max_dist=10000):
        route = a_star(graph=self.graph,
                       nodes=self.nodes,
                       src=src,
                       dst=dst,
                       search_memo=self.search_memo,
                       nodes_dist_memo=self.nodes_dist_memo,
                       max_steps=max_steps,
                       max_dist=max_dist
        )
       
        return route

    
    def spatial_query(self, geofence, name='df_edges', predicate='intersects'):
        gdf = getattr(self, name)
        
        return gdf.sindex.query(geofence, predicate=predicate)


    """ get attribute """
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


    """ transfrom """
    def transform_node_seq_to_df_edge(self, node_lst:list, on:list=['src', 'dst'], attrs:list=None):
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


    def transform_node_seq_to_polyline(self, node_lst:list):
        """Create Linestring by node id sequence.

        Args:
            path (list): The id sequence of Coordinations.
            net (Digraph_OSM): The Digraph_OSM object.

        Returns:
            Linestring: The linstring of the speicla sequence.
        """
        if node_lst is None or len(node_lst) <= 1 or np.isnan(node_lst).all():
            return None
        
        steps = self.transform_node_seq_to_df_edge(node_lst, attrs=['geometry'])
        coords = np.concatenate(steps.geometry.apply(lambda x: x.coords), axis=0)
        
        return LineString(coords)       
        

    """ io """
    def to_postgis(self, name):
        try:
            df_node_with_degree = self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()
            
            gdf_to_postgis(self.df_edges, f'topo_osm_{name}_edge')
            gdf_to_postgis(df_node_with_degree, f'topo_osm_{name}_endpoint')
            
            self.df_nodes.loc[:, 'nid'] = self.df_nodes.index
            self.df_nodes = self.df_nodes[['nid', 'x', 'y', 'traffic_signals', 'geometry']]
            gdf_to_postgis(self.df_nodes, f'topo_osm_{name}_node')
            return True
        except:
            if self.logger is not None:
                self.logger.error('upload data error.')
        
        return False


    def to_csv(self, name, folder = None):
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


    def save_checkpoint(self, ckpt):
        return save_checkpoint(self, ckpt)


    def load_checkpoint(self, ckpt):
        return load_checkpoint(ckpt, self)
    

if __name__ == "__main__":
    network = GeoDigraph()
    network.load_checkpoint(ckpt='/home/pcl/codes/ST-MapMatching/cache/Shenzhen_graph.ckpt')

    path = network.search(src=7959990710, dst=499265789)
    from tilemap import plot_geodata
    plot_geodata(network.df_nodes.loc[path['path']])

    # TODO od 间的 geometry
    network.transform_node_seq_to_df_edge(path['path'])

