import os
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from haversine import haversine, Unit
from shapely.geometry import Point, LineString

from .coord.coordTransfrom_shp import coord_transfer 
from utils.interval_helper import merge_intervals
from utils.timer import Timer
from db.db_process import gdf_to_geojson


def download_osm_xml(fn, bbox, verbose=False):
    """Download OSM map of bbox from Internet.

    Args:
        fn (function): [description]
        bbox ([type]): [description]
        verbose (bool, optional): [description]. Defaults to False.
    """
    if os.path.exists(fn):
        return True

    if verbose:
        print("Downloading {}".format(fn))
    
    if isinstance(bbox, list) or isinstance(bbox, np.array):
        bbox = ",".join(map(str, bbox))

    try:
        import requests
        url = f'http://overpass-api.de/api/map?bbox={bbox}'
        r = requests.get(url, stream=True)
        with open(fn, 'wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)

        if verbose:
            print("Downloaded success.\n")

        return True
    except:
        return False


def parse_xml_to_topo(fn, road_info_fn, type_filter=[], keep_cols=None, crs=4326, in_sys='wgs', out_sys='wgs'):
    """OSM XML parser: extract topo

    Args:
        fn (Path): OSM XML path.
        road_info_fn (Path): The additional road infomation file, e.g., Speed.
        type_filter (list, optional): Road type Filter. Defaults to [].
        keep_cols (list, optional): The remainings attrs of link, if None then filter nothing. Defaults to None.
        crs(int, optional): The crs of GeoDataFrame. Default to '4326'
        in_sys (str, optional): Input coordination system. Defaults to 'wgs'.
        out_sys (str, optional): Output coordination system. Defaults to 'wgs'.

    Returns:
        (nodes, links) (gpd.GeoDataFrame, gpd.GeoDataFrame)
    """
    import xml
    from xml.dom import minidom

    dom      = xml.dom.minidom.parse(fn)
    root     = dom.documentElement
    nodelist = root.getElementsByTagName('node')
    waylist  = root.getElementsByTagName('way')

    def _node_parser():
        nodes = []
        for node in tqdm(nodelist, 'Parse nodes \t'):
            pid = node.getAttribute('id')
            taglist = node.getElementsByTagName('tag')
            info = {
                'pid': int(pid),
                'y'  :float(node.getAttribute('lat')), 
                'x'  :float(node.getAttribute('lon'))
            }
            
            for tag in taglist:
                if tag.getAttribute('k') == 'traffic_signals':
                    info['traffic_signals'] = tag.getAttribute('v')
            
            nodes.append(info)
                    
        nodes = pd.DataFrame(nodes)
        df_nodes = gpd.GeoDataFrame(
            nodes,
            geometry=nodes.apply(lambda i: Point(i.x, i.y), axis=1),
            crs=f"EPSG:{crs}" 
        ).set_index('pid')

        if in_sys != out_sys:
            df_nodes = coord_transfer(df_nodes, in_sys, out_sys)
            df_nodes.loc[:,['x']], df_nodes.loc[:,['y']] = df_nodes.geometry.x, df_nodes.geometry.y

        return df_nodes

    def _edge_parser(nodes):
        edges = []
        for way in tqdm(waylist, 'Parse edges\t'):
            taglist = way.getElementsByTagName('tag')
            info = {tag.getAttribute('k'): tag.getAttribute('v') for tag in taglist}
            if 'highway' not in info or info['highway'] in type_filter:
                continue
            
            info['rid'] = int(way.getAttribute('id'))
            ndlist = way.getElementsByTagName('nd')
            nds = []
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                nds.append(nd_id)
            for i in range(len(nds)-1):
                edges.append( { 'order': i, 's':nds[i], 'e':nds[i+1], 'road_type': info['highway'], **info} )

        edges = pd.DataFrame(edges)
        edges.loc[:, ['s','e']] = pd.concat((edges.s.astype(np.int), edges.e.astype(np.int)), axis=1)

        edges = edges.merge( nodes[['x','y']], left_on='s', right_index=True ).rename(columns={'x':'x0', 'y':'y0'}) \
                        .merge( nodes[['x','y']], left_on='e', right_index=True ).rename(columns={'x':'x1', 'y':'y1'})
        edges = gpd.GeoDataFrame(
            edges, 
            geometry=edges.apply( lambda i: LineString( [[i.x0, i.y0], [i.x1, i.y1]] ), axis=1 ),
            crs=f"EPSG:{crs}" 
        )
        edges.loc[:, 'dist'] = edges.apply(lambda i: haversine((i.y0, i.x0), (i.y1, i.x1), unit=Unit.METERS), axis=1)
        edges.sort_values(['rid', 'order'], inplace=True)

        return edges

    nodes = _node_parser()
    edges = _edge_parser(nodes)
    nodes = nodes.loc[np.unique(np.hstack((edges.s.values, edges.e.values))), :] # nodes filter

    if road_info_fn and os.path.exists(road_info_fn):
        road_speed = pd.read_csv(road_info_fn)[['road_type', 'v']]
        edges = edges.merge(road_speed, on ='road_type')
    
    keep_cols = edges.columns if keep_cols is None else [i for i in keep_cols if i in edges.columns]
    
    return nodes, edges[keep_cols]


def combine_links(rid, edges_, omit_nids):
    """Combine OSM links with `rid`.

    Args:
        rid (int): The id of link in OSM.
        nodes (gdf.GeoDataFrame): The all nodes related to `rid` road.
        links (gdf.GeoDataFrame): The all links related to `rid` road. 
        omit_points (list): The points don't meet: 1) only has 1 indegree and 1 outdegree; 2) not the traffic_signals point.

    Returns:
        gpd.GeoDataFrame: The links after combination.
    """
    edges_.loc[:, 'waypoints'] = edges_.apply(lambda x: f"{x.s},{x.e}",axis=1)
    if len(omit_nids) == 0:
        return edges_
    
    # assert 'order' in list(edges_) + [edges_.index.name], f"check rid: {rid}"
    # edges_ = edges.copy()
    # nodes.drop_duplicates(inplace=True)
    
    if 'order' in edges_.columns:
        edges_.set_index('order', inplace=True)
    combine_seg_indxs = merge_intervals([[x-1, x] for x in omit_nids if x > 0])

    drop_index = []
    for start, end, _ in combine_seg_indxs:
        segs = edges_.query(f"{start} <= order <= {end}")
        nids = np.append(segs.s.values, segs.iloc[-1]['e'])

        edges_.loc[start, 'e']         = segs.iloc[-1]['e']
        edges_.loc[start, 'dist']      = segs.dist.sum()
        # edges_.loc[start, 'geometry']  = LineString([[nodes.loc[p]['x'], nodes.loc[p]['y']] for p in nids])
        edges_.loc[start, "waypoints"] = ",".join((str(i) for i in nids))

        drop_index += [ i for i in range(start+1, end+1) ]

    edges_.drop(index=drop_index, inplace=True)
    edges_.reset_index(inplace=True)
    
    return edges_


def combine_links_parallel_helper(df_tuple, nodes, omit_pid_dict, verbose=False):
    i, df = df_tuple
    if verbose:
        timer = Timer()
        timer.start()
        print(f"Part {i} start, size: {df.shape[0]}")

    res = pd.concat([combine_links(i, df, omit_pid_dict.get(i, [])) for i, df in df.groupby('rid')])
    res = gpd.GeoDataFrame(
        res, 
        geometry=res.waypoints.apply(
            lambda x: LineString( nodes.loc[int(i)].geometry.coords[0] for i in x.split(',')) 
        )
    )
        
    if verbose: print(f"Part {i} Done, {timer.stop():.2f} s")
    
    return res


if __name__ == "__main__":
    download_osm_xml('pcl', [113.934529 ,  22.5753099, 113.9369767,  22.5753355])