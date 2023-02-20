import os
os.environ["USE_PYGEOS"] = "1"

from ..graph import GeoDigraph
from ..osmnet.downloader import download_osm_xml
from ..osmnet.parse_osm_xml import parse_xml_to_graph
from ..setting import DATA_FOLDER


def load_geograph(ckpt, ll):
    graph = GeoDigraph()
    graph.load_checkpoint(ckpt)
    graph.init_searcher()

    if not ll:
        graph.to_proj()

    return graph


def build_geograph(xml_fn=None, bbox=None, ckpt=None, ll=False, way_info=True, upload_to_db=False, *args, **kwargs):
    if ckpt:
        return load_geograph(ckpt, ll)

    if not os.path.exists(xml_fn):
        download_osm_xml(xml_fn, bbox, False)
    
    df_nodes, df_edges, df_ways = parse_xml_to_graph(xml_fn, *args, **kwargs)
    
    graph = GeoDigraph(df_edges, df_nodes, ll=ll)
    if not ll:
        graph.to_proj()

    return graph

if __name__ == "__main__":
    # new graph
    name = 'GBA'
    graph = build_geograph(xml_fn = f"../../data/network/{name}.osm.xml")
    graph.save_checkpoint(f'../../data/network/{name}_graph_pygeos.ckpt')
    
    # load ckpt
    graph = build_geograph(ckpt=f'../../data/network/{name}_graph_pygeos.ckpt')
    # check
    path = graph.search(src=7959990710, dst=499265789)
    graph.get_edge(path['epath']).plot()

    # save to DB
    # graph.to_postgis(name)
    
    import networkx