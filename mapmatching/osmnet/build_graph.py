import os
os.environ["USE_PYGEOS"] = "1"

from ..graph import GeoDigraph
from ..osmnet.downloader import download_osm_xml
from ..osmnet.parse_osm_xml import parse_xml_to_graph
from ..setting import DATA_FOLDER


def load_geograph(ckpt):
    graph = GeoDigraph()
    graph.load_checkpoint(ckpt)

    return graph


def build_geograph(xml_fn=None, bbox=None, ckpt=None, way_info=True, *args, **kwargs):
    if ckpt:
        return load_geograph(ckpt)

    download_osm_xml(xml_fn, bbox, False)
    df_nodes, df_edges, df_ways = parse_xml_to_graph(xml_fn, *args, **kwargs)
    
    attrs = ['highway', 'name', 'maxspeed', 'oneway', 'lanes']
    if way_info: 
        df_edges = df_edges.merge(df_ways[attrs], left_on='way_id', right_index=True)
    graph = GeoDigraph(df_edges, df_nodes)
    
    return graph
    

if __name__ == "__main__":
    # new graph
    name = 'Shenzhen'
    graph = build_geograph(xml_fn = f"../../data/network/{name}.osm.xml")
    graph.save_checkpoint(f'../../data/network/{name}_graph_12_pygeos.ckpt')
    
    # load ckpt
    graph = build_geograph(ckpt=f'../../data/network/{name}_graph_12_pygeos.ckpt')
    
    # check
    path = graph.search(src=7959990710, dst=499265789)
    graph.get_edge(path['path']).plot()

    # save to DB
    # from utils.db import gdf_to_postgis
    # gdf_to_postgis(graph.df_edges.rename(columns={'highway': 'road_type'}), f'topo_osm_{name}_edge')
    # gdf_to_postgis(graph.df_nodes, f'topo_osm_{name}_node')
    