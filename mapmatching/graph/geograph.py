import os
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString
# from networkx.classes import DiGraph

from .base import Digraph
from .astar import Astar
from .bi_astar import Bi_Astar
from ..osmnet.twoway_edge import parallel_offset_edge, swap_od
from ..utils.serialization import save_checkpoint, load_checkpoint


class GeoDigraph(Digraph):
    def __init__(self, df_edges:GeoDataFrame=None, df_nodes:GeoDataFrame=None, *args, **kwargs):
        self.df_edges = df_edges
        self.df_nodes = df_nodes
        self.search_memo = {}
        self.nodes_dist_memo = {}

        if df_edges is not None and df_nodes is not None:
            super().__init__(df_edges[['src', 'dst', 'dist']].sort_index().values, 
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
        
        if 'epath' not in route:
            route['epath'] = self.transform_vpath_to_epath(route['vpath'])
        
        if geom and 'geometry' not in route:
            lst = route['epath']
            if lst is None:
                route['geometry'] = None
            else:
                route['geometry'] = self.transform_epath_to_linestring(lst)
        
        return route

    def spatial_query(self, geofence, name='df_edges', predicate='intersects'):
        gdf = getattr(self, name)
        
        idxs = gdf.sindex.query(geofence, predicate=predicate)
        
        return gdf.iloc[idxs].index

    """ get attribute """
    def get_edge(self, eid, attrs=None, reset_index=False):
        """Get edge by eid [0, n]"""
        res = self._get_feature('df_edges', eid, attrs, reset_index)

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

    def get_pred_edges(self, eid):
        src, _ = self.eid_2_od[eid]
        eids = [i['eid'] for i in self.graph_r[src].values()]

        return self.df_edges.loc[eids]

    def get_succ_edges(self, eid):
        _, dst = self.eid_2_od[eid]
        eids = [i['eid'] for i in self.graph[dst].values()]

        return self.df_edges.loc[eids]

    def get_way(self, way_id):
        df = self.df_edges.query("way_id == @way_id")
        if df.shape[0] == 0:
            return None
        
        return df

    """ transfrom """
    def transform_epath_to_linestring(self, eids):
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

    """ adding and removing nodes and edges """
    def add_edge(self, start, end, length=None):
        # add edge to dataframe
        return super().add_edge(start, end, length)
    
    def add_reverse_way(self, way_id, od_attrs=['src', 'dst'], offset=True):
        df_edges_rev = self.df_edges.query('way_id == @way_id')
        if df_edges_rev.shape[0] == 0:
            print(f"check way id {way_id} exist or not.")
            return False
        if df_edges_rev.dir.nunique() >= 2:
            return False

        ring_mask = df_edges_rev.geometry.apply(lambda x: x.is_ring)
        df_edges_rev = df_edges_rev[~ring_mask]
        df_edges_rev = swap_od(df_edges_rev)

        self.add_edges_from_df(df_edges_rev)

        # two ways offsets
        if offset:
            idxs = self.df_edges.query('way_id == @way_id').index
            self.df_edges.loc[idxs, 'geom_origin'] = self.df_edges.loc[idxs].geometry.copy()
            self.df_edges.loc[idxs, 'geometry'] = self.df_edges.loc[idxs].apply( lambda x: parallel_offset_edge(x), axis=1 )

        return True

    def add_edges_from(self, edges):
        return super().build_graph(edges)

    def add_edges_from_df(self, df):
        eids = range(self.max_eid, self.max_eid + df.shape[0])
        df.index = df.loc[:, 'eid'] = eids

        df_edges = gpd.GeoDataFrame(pd.concat([self.df_edges, df]))
        self.df_edges = df_edges

        return self.add_edges_from(df[['src', 'dst', 'dist']].values) 

    def remove_edge(self, eid=None, src=None, dst=None):
        assert eid is not None or src is not None and dst is not None

        if eid is None:
            eid = self.get_eid(src, dst)
        else:
            src, dst = self.eid_2_od[eid]

        self.df_edges.drop(index=eid, inplace=True)
        self.search_memo.clear()

        return super().remove_edge(src, dst)

    def split_edge(self, eid):
        # TODO 
        # !  delete multi-edges 
        idxs = check_multi_edges(net)

        keep_idx = df_edges.loc[eid]\
                            .sort_values(['level', 'dist'], ascending=[True, True])\
                            .groupby(['src', 'dst'])\
                            .head(1).index
        remove_idx = [i for i in idxs if i not in keep_idx]

        for id in remove_idx:
            print(f"Split multi-edge: {id}")
            edges = net.get_edge(remove_idx)
            net.remove_edge(id)

            waypoints = edges.iloc[0].waypoints
            coords = edges.iloc[0].geometry.coords[:]
            tmp = pd.concat([edges] * (len(waypoints) - 1)).reset_index(drop=True)

            tmp.loc[:, 'order'] = tmp.index
            tmp.loc[:, 'src'] = waypoints[:-1]
            tmp.loc[:, 'dst'] = waypoints[1:]
            tmp.loc[:, 'waypoints'] = tmp.apply(lambda x: x.waypoints[x.name: x.name + 2], axis=1)
            tmp.loc[:, 'geometry'] = tmp.apply(lambda x: LineString(coords[x.name: x.name + 2]), axis=1)

            net.add_edges_from_df(tmp)
            
        return NotImplementedError


if __name__ == "__main__":
    network = GeoDigraph()
    network.load_checkpoint(ckpt='./cache/Shenzhen_graph_9.ckpt')

    route = network.search(src=7959990710, dst=499265789)
    # route = network.search(src=7959602916, dst=7959590857)

    network.df_edges.loc[route['epath']].plot()
    
    network.transform_epath_to_linestring(route['epath'])
