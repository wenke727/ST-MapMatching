import os
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


def build_geograph(xml_fn:str=None, bbox:list=None, ckpt:str=None, 
                   ll=False, *args, **kwargs):
    """Build geograph by one of the three type: 1) xml_fn, 2) bbox, 3) ckpt. 
    The prior is: ckpt > xml_fn > bbox

    Args:
        xml_fn (str, optional): Local OSM network file path. When `xml_fn` is not exist
        and the `bbox` is config, the OSM file would be downloaded and save at that location. Defaults to None.
        bbox (list, optional): Download the OSM netkork by the Bounding box. Defaults to None.
        ckpt (str, optional): Checkpoint. Defaults to None.
        ll (bool, optional): Use lon/lat coordination system. Defaults to False.

    Returns:
        GeoDigraph: graph
    """
    assert xml_fn is not None or bbox is not None or ckpt is not None
    if ckpt:
        return load_geograph(ckpt, ll)

    if not os.path.exists(xml_fn):
        assert bbox is not None, \
              "The local osm file is not exists, please config bbox to dowload"
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