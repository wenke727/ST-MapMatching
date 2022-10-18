import sys

sys.path.append('..')

from graph import GeoDigraph

from osmnet.downloader import download_osm_xml
from osmnet.parse_osm_xml import parse_xml_to_graph
from osmnet.osm_io import load_graph, save_graph


def load_geograph(ckpt):
    graph = GeoDigraph()
    graph.load_checkpoint(ckpt)

    return graph


def build_geograph(xml_fn=None, bbox=None, ckpt=None, *args, **kwargs):
    if ckpt:
        return load_geograph(ckpt)

    # TODO df_ways
    download_osm_xml(xml_fn, bbox, False)
    df_nodes, df_edges, df_ways = parse_xml_to_graph(xml_fn, *args, **kwargs)
    graph = GeoDigraph(df_edges, df_nodes)
    
    return graph
    

if __name__ == "__main__":
    graph = build_geograph("../../cache/Shenzhen.osm.xml")

    path = graph.search(src=7959990710, dst=499265789)
    graph.df_nodes.loc[path['path']].plot()

    graph.save_checkpoint('../../cache/Shenzhen.graph.ckpt')
