import sys
sys.path.append('..')

from graph import GeoDigraph
from osmnet.downloader import download_osm_xml
from osmnet.parse_osm_xml import parse_xml_to_graph


def load_geograph(ckpt):
    graph = GeoDigraph()
    graph.load_checkpoint(ckpt)

    return graph


def build_geograph(xml_fn=None, bbox=None, ckpt=None, way_info=True, *args, **kwargs):
    if ckpt:
        return load_geograph(ckpt)

    # TODO df_ways
    download_osm_xml(xml_fn, bbox, False)
    df_nodes, df_edges, df_ways = parse_xml_to_graph(xml_fn, *args, **kwargs)
    
    attrs = ['highway', 'name', 'maxspeed', 'oneway', 'lanes']
    if way_info: 
        df_edges = df_edges.merge(df_ways[attrs], left_on='way_id', right_index=True)
    graph = GeoDigraph(df_edges, df_nodes)
    
    return graph
    

if __name__ == "__main__":
    # new graph
    graph = build_geograph("../../cache/Shenzhen.osm.xml")
    graph.save_checkpoint('../../cache/Shenzhen_graph_9.ckpt')
    
    # load ckpt
    # graph = build_geograph(ckpt='../../cache/Shenzhen_graph_9.ckpt')
    
    # check
    path = graph.search(src=7959990710, dst=499265789)
    graph.df_nodes.loc[path['path']].plot()

    # save to DB
    # gdf_to_postgis(graph.df_edges.rename(columns={'highway': 'road_type'}), 'topo_osm_shenzhen_edge')
    # gdf_to_postgis(graph.df_nodes, 'topo_osm_shenzhen_node')
    