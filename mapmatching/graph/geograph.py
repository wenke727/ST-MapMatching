import os
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from .base import Digraph
from .astar import Astar, Bi_Astar
from ..utils.serialization import save_checkpoint, load_checkpoint


class GeoDigraph(Digraph):
    def __init__(self, df_edges:GeoDataFrame=None, df_nodes:GeoDataFrame=None, *args, **kwargs):
        self.df_edges = df_edges
        self.df_nodes = df_nodes
        self.search_memo = {}
        self.nodes_dist_memo = {}

        if df_edges is not None and df_nodes is not None:
            super().__init__(df_edges[['src', 'dst', 'dist']].values, 
                             df_nodes.to_dict(orient='index'), *args, **kwargs)
            self.init_searcher()

    def init_searcher(self, algs='astar'):
        if algs == 'astar':
            self.searcher = Astar(self.graph, self.nodes, 
                                search_memo=self.search_memo, 
                                nodes_dist_memo=self.nodes_dist_memo,
                                max_steps=2000, max_dist=10000)
        else:
            self.searcher = Bi_Astar(self.graph, self.graph_r, self.nodes,
                                    search_memo=self.search_memo,
                                    nodes_dist_memo=self.nodes_dist_memo
                                    )

        
    def search(self, src, dst, max_steps=2000, max_dist=10000, geom=True):
        route = self.searcher.search(src, dst, max_steps, max_dist)
        
        if 'path' not in route:
            route['path'] = self.transform_node_seq_to_edge_seq(route['waypoints'])
        
        if geom and 'geometry' not in route:
            lst = route['path']
            if lst is None:
                route['geometry'] = None
            else:
                route['geometry'] = self.transform_edge_seq_to_polyline(lst)
        
        return route

    
    def spatial_query(self, geofence, name='df_edges', predicate='intersects'):
        gdf = getattr(self, name)
        
        idxs = gdf.sindex.query(geofence, predicate=predicate)
        
        return gdf.iloc[idxs].index


    """ get attribute """
    def get_edge(self, eid, attrs=None, reset_index=False):
        """Get edge by eid [0, n]"""
        res = self._get_feature('df_edges', eid, attrs, reset_index)
        if attrs == 'waypoints':
            res = [int(i) for i in res.split(",")]
            
            return res
        
        return res


    def get_node(self, nid, attrs=None, reset_index=False):
        return self._get_feature('df_nodes', nid, attrs, reset_index)


    def _get_feature(self, df_name, id, attrs=None, reset_index=False):
        """get edge by id.

        Args:
            id (_type_): _description_
            att (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        df = getattr(self, df_name)
        if df is None:
            print(f"Don't have the attibute {df_name}")
            return None

        if isinstance(id, int) and id not in df.index:
            return None

        if isinstance(id, list) or isinstance(id, tuple) or isinstance(id, np.ndarray):
            for i in id:
                if i not in df.index:
                    return None
        
        res = df.loc[id]
        
        if attrs is not None:
            res = res[attrs]
        
        if reset_index:
            res.reset_index(drop=True, inplace=True)
        
        return res


    """ transfrom """
    def transform_node_seq_to_edge_seq(self, node_lst:np.array, on:list=['src', 'dst'], key='eid'):
        if node_lst is None or len(node_lst) <= 1:
            return None
        
        df = pd.DataFrame({on[0]: node_lst[:-1], 
                           on[1]: node_lst[1:]})
        
        # Tips: the merge operation sequence is important
        eids = df.merge(self.df_edges[on + [key]], on=on)[key].values.tolist()

        return eids


    def transform_edge_seq_to_polyline(self, eids):
        steps = self.get_edge(eids, attrs=['geometry'], reset_index=True)
        coords = np.concatenate(steps.geometry.apply(lambda x: x.coords), axis=0)
        
        return LineString(coords)           
           

    """ io """
    def to_postgis(self, name, nodes_attrs=['nid', 'x', 'y', 'traffic_signals', 'geometry']):
        from ..utils.db import gdf_to_postgis
        df_node_with_degree = self.df_nodes.merge(self.calculate_degree(), left_index=True, right_index=True).reset_index()
        
        gdf_to_postgis(self.df_edges, f'topo_osm_{name}_edge')
        gdf_to_postgis(df_node_with_degree, f'topo_osm_{name}_endpoint')
        
        self.df_nodes.loc[:, 'nid'] = self.df_nodes.index
        nodes_attrs = [i for i in nodes_attrs if i in list(self.df_nodes) ]
        self.df_nodes = self.df_nodes[nodes_attrs]
        gdf_to_postgis(self.df_nodes, f'topo_osm_{name}_node')

        return True


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
        from loguru import logger
        
        load_checkpoint(ckpt, self)
        self.logger = logger

        return self


if __name__ == "__main__":
    network = GeoDigraph()
    network.load_checkpoint(ckpt='./cache/Shenzhen_graph_9.ckpt')

    route = network.search(src=7959990710, dst=499265789)
    # route = network.search(src=7959602916, dst=7959590857)

    from tilemap import plot_geodata
    plot_geodata(network.df_edges.loc[route['path']])
    
    network.transform_edge_seq_to_polyline(route['path'])
