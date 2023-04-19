#%%
from tqdm import tqdm
import networkx as nx
import itertools
import geopandas as gpd
from networkx import shortest_simple_paths
from pathlib import Path

from stmm import build_geograph, ST_Matching
from tilemap import plot_geodata

"""step 1: 获取/加载路网"""
folder = Path("./data/network")
# 方法1：
# 根据 bbox 从 OSM 下载路网，从头解析获得路网数据
# net = build_geograph(bbox = [113.928518,  22.551085, 114.100451,  22.731744],
#                      xml_fn = folder / "SZN.osm.xml", ll=False, n_jobs=32)
# 将预处理路网保存为 ckpt
# net.save_checkpoint(folder / 'SZN_graph.ckpt')

# net = build_geograph(ckpt='../dataset/cache/SZN_graph.ckpt') 
net = build_geograph(ckpt = folder / 'SZN_graph.ckpt') 
matcher = ST_Matching(net=net, ll=False)

#%%
plot_geodata(net.df_edges.to_crs(4326))
net.df_edges.head(5)

# %%

def get_k_shortest_paths(G, u, v, k):
    paths_gen = shortest_simple_paths(G, u, v, "length")
    for path in itertools.islice(paths_gen, 0, k):
        yield path

def plot_top_k_shortest_path():
    geoms = []

    for path in get_k_shortest_paths(G, 9168697035, 9167366553, 3):
        epath = net.transform_vpath_to_epath(path)
        path_geom = net.transform_epath_to_linestring(epath)
        geoms.append(path_geom)

    geoms = gpd.GeoDataFrame(geometry=geoms, crs=net.df_edges.crs)

    plot_geodata(geoms.to_crs(4326).reset_index(), column='index', legend=True, alpha=.5)

    return 

G = nx.DiGraph()

# # 最短路测试
# nx.shortest_path(G, 9168697035, 9167366553, weight='dist')


#%%
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd

class GeoGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, reindex_node=True, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.nodeid_long2short = {}
        self.nodeid_short2long = {}
        self.nxt_nid = 0
        self.reindex_node = reindex_node

    def search(self, o, d):
        return nx.shortest_path(self, o, d, weight='weight')

    def load_graph(self, edges:gpd.GeoDataFrame, src='src', dst='dst',  weight='dist'):
        # 新增边
        for name, item in tqdm(edges.iterrows()):
            o = item[src]
            d = item[dst]

            if self.reindex_node:
                o = self._get_short_node_id(o)
                d = self._get_short_node_id(d)

            _w = item[weight]
            self.add_edge(o, d, weight=_w)

    def _get_short_node_id(self, nid):
        if not self.reindex_node:
            return nid
        
        if nid in self.nodeid_long2short:
            return self.nodeid_long2short[nid]
        
        self.nodeid_long2short[nid] = self.nxt_nid
        self.nodeid_short2long[self.nxt_nid] = nid
        tmp = self.nxt_nid
        self.nxt_nid += 1
        
        return tmp

    """ coordination """
    def align_crs(self, gdf):
        return

    """ vis """
    def add_edge_map(self, ax, *arg, **kwargs):
        return
        
    """ property"""
    @property
    def crs(self):
        return self.df_edges.crs
    
    @property
    def epsg(self):
        return self.df_edges.crs.to_epsg()


digraph = GeoGraph(reindex_node=False)
digraph.load_graph(net.df_edges)
o, d = 9168697035, 9167366553

o = digraph._get_short_node_id(o)
d = digraph._get_short_node_id(d)
digraph.search(o, d)


# %%
