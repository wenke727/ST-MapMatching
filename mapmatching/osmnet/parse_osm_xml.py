#%%
import re
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
from osmium import SimpleHandler
from shapely.geometry import Point, LineString

warnings.filterwarnings('ignore')

from ..utils.timer import Timer, timeit
from ..osmnet.misc import Bunch, cal_od_straight_distance
from ..osmnet.twoway_edge import add_reverse_edge, swap_od, edge_offset
from ..osmnet.combine_edges import pipeline_combine_links, parallel_process
from ..setting import highway_filters, link_type_level_dict
from ..geo.ops.to_array import points_geoseries_2_ndarray


class WayHandler(SimpleHandler):
    def __init__(self, link_type_filter=None):
        SimpleHandler.__init__(self)
        self.link_type_filter = link_type_filter
        
        self.way_nodes = set()
        self.osm_way_dict = {}
        
        self.edges = []

    def way(self, w):
        way = Bunch()
        way['highway'] = w.tags.get('highway')

        if way.highway is None:
            return
        if self.link_type_filter is not None and way.highway in self.link_type_filter:
            return

        # ways # TODO filter key params
        way['osm_way_id'] = int(w.id)
        way['ref_node_id_list'] = [int(node.ref) for node in w.nodes]
        way['name'] = w.tags.get('name')
        way['service'] = w.tags.get('service')
        self._extract_lane_info(w, way)
        self._extract_maxspeed_info(w, way)
        self._extract_one_way_info(w, way)
        self.osm_way_dict[way['osm_way_id']] = way

        self.way_nodes.update(way.ref_node_id_list)

        # edges
        nds = way.ref_node_id_list
        for i in range(len(way.ref_node_id_list) - 1):
            item = {'way_id': way['osm_way_id'], 
                    'order': i, 
                    'src': nds[i], 
                    'dst': nds[i+1],
                    'waypoints': f"[{nds[i]},{nds[i+1]}]" 
            }
            self.edges.append(item)

    def _extract_one_way_info(self, w, way):
        # TODO check and update one way info by the `detour coefficient`
        oneway_info = w.tags.get('oneway')
        way['oneway'] = None
        if oneway_info is not None:
            if oneway_info == 'yes' or oneway_info == '1':
                way['oneway'] = True
            elif oneway_info == 'no' or oneway_info == '0':
                way['oneway'] = False
            elif oneway_info == '-1':
                way['oneway'] = True
                way.is_reversed = True
            elif oneway_info in ['reversible', 'alternating']:
                # reversible, alternating: https://wiki.openstreetmap.org/wiki/Tag:oneway%3Dreversible
                way['oneway'] = False
            else:
                logger.warning(f'new lane type detected at way {way.osm_way_id}, {oneway_info}')   
        else:
            way['oneway'] = True
        
        return way

    def _extract_lane_info(self, w, way):
        lane_info = w.tags.get('lanes')
        if lane_info is not None:
            lanes = re.findall(r'\d+\.?\d*', lane_info)
            if len(lanes) > 0:
                way['lanes'] = int(float(lanes[0]))  # in case of decimals
            else:
                logger.warning(f"new lanes type detected at way {way['osm_way_id']}, {lane_info}")
        lane_info = w.tags.get('lanes:forward')
        if lane_info is not None:
            try:
                way['forward_lanes'] = int(lane_info)
            except:
                pass
        lane_info = w.tags.get('lanes:backward')
        if lane_info is not None:
            try:
                way['backward_lanes'] = int(lane_info)
            except:
                pass        

        return way

    def _extract_maxspeed_info(self, w, way):
        maxspeed_info = w.tags.get('maxspeed')
        way['maxspeed'] = None
        if maxspeed_info is not None:
            try:
                way['maxspeed'] = int(float(maxspeed_info))
            except ValueError:
                speeds = re.findall(r'\d+\.?\d* mph', maxspeed_info)
                if len(speeds) > 0:
                    way['maxspeed'] = int(float(speeds[0][:-4]) * 1.6)
                else:
                    speeds = re.findall(r'\d+\.?\d* km/h', maxspeed_info)
                    if len(speeds) > 0:
                        way['maxspeed'] = int(float(speeds[0][:-5]))
                    else:
                        logger.warning(f'new maxspeed type detected at way {way.osm_way_id}, {maxspeed_info}')                            

class NodeHandler(SimpleHandler):
    def __init__(self, node_filter=None):
        SimpleHandler.__init__(self)
        self.nodes = {}
        self.node_filter = node_filter


    def node(self, n):
        osm_node_id = int(n.id)
        if self.node_filter is not None and osm_node_id not in self.node_filter:
            return None
        
        lon, lat = n.location.lon, n.location.lat
        node_geometry = Point(lon, lat)
        # in_region = False if self.strict_mode and (not node_geometry.within(self.bounds)) else True

        osm_node_name = n.tags.get('name')
        osm_highway = n.tags.get('highway')
        ctrl_type = 'signal' if (osm_highway is not None) and 'signal' in osm_highway else None

        item = {'nid': osm_node_id,
                'name': osm_node_name, 
                'highway': osm_highway, 
                'ctrl_type': ctrl_type, 
                "x": lon,
                "y": lat,
                "xy": (lon, lat),
                'geometry': node_geometry
        }
        
        self.nodes[osm_node_id] = item

def check_multi_edges(df_edges):
    tmp = df_edges.groupby(['src', 'dst']).apply(lambda x: x.index.values)
    idxs = tmp[tmp.apply(len) > 1].values
    if len(idxs) < 1:
        return []
        
    idxs = set(np.concatenate(idxs))
    if len(idxs):
        logger.debug(f"Exists {len(idxs)} multi-egdes: \n\t{idxs}")
        return list(idxs)
    
    return []

def __extract_waypoints_2_od(points):
    if isinstance(points, str):
        points = eval(points)
    assert isinstance(points, list), "check od type"
    
    return [str([points[i], points[i+1]]) for i in range(len(points)-1)]

def __check_dulplicates(df):
    # BUG
    src, dst = 10253667641, 10253667620
    tmp = df.query('src == @src and dst == @dst')
    if tmp.shape[0] > 1:
        print("Exist duplicates !!!!")

def post_process_ways(osm_way_dict):
    df_ways = pd.DataFrame.from_dict(osm_way_dict, orient='index')
    df_ways.loc[:, 'link'] = df_ways.highway.apply(lambda x: "link" in x)
    df_ways.loc[:, 'highway'] = df_ways.highway.apply(lambda x: x.split('_')[0])
    df_ways.loc[:, 'level'] = df_ways.highway.apply(lambda x: link_type_level_dict.get(x, 99))
    df_ways.loc[:, 'src'] = df_ways.ref_node_id_list.apply(lambda x: x[0])
    df_ways.loc[:, 'dst'] = df_ways.ref_node_id_list.apply(lambda x: x[-1])
    assert df_ways.oneway.nunique() >= 2, "check oneways"
    
    return df_ways

def drop_duplicates_way(df, df_ways=None, att="ref_node_id_list", sort_atts=['src', 'dst', 'level', 'order']):
    lst = df.loc[:, att].apply(str)
    mask = lst.value_counts() > 1
    contents = mask[mask].index

    idxs = lst.to_frame().query(f"{att} in @contents").index
    _df = df.loc[idxs]
    if df_ways:
        _df = append_way_info(_df, df_ways, ['level'])

    duplicates = _df.sort_values([i for i in sort_atts if i in list(df)])
    keep_idxs = set(duplicates.groupby(['src', 'dst']).head(1).index)
    drop_idxs = set([i for i in duplicates.index if i not in keep_idxs])

    df.drop(index=drop_idxs, inplace=True)

    return df, keep_idxs, drop_idxs

def append_way_info(df_edges, df_ways, way_attrs=None):
    if way_attrs is None:
        way_attrs = list(df_ways)
    
    df_edges = df_edges.merge(df_ways[way_attrs], left_on='way_id', right_index=True, how='left')
    if "highway" in way_attrs:
        df_edges.rename(columns={'highway': 'road_type'}, inplace=True)
    
    return df_edges

@timeit
def _parse_xml(fn, highway_filters, crs="epsg:4326", in_sys='wgs', out_sys='wgs'):
    # ways
    timer = Timer()
    way_handler = WayHandler(highway_filters)
    way_handler.apply_file(fn)
    df_ways = post_process_ways(way_handler.osm_way_dict)
    df_ways, keep_idxs, drop_idxs = drop_duplicates_way(df_ways)    
    print(f"Parse ways: {timer.stop():.2f} s")
    logger.debug(f"\tDrop duplicates ways, way ids: {list(drop_idxs)}")

    # nodes 
    timer.start()
    node_handler = NodeHandler(way_handler.way_nodes)
    node_handler.apply_file(fn)
    df_nodes = gpd.GeoDataFrame.from_dict(node_handler.nodes, orient='index', crs=crs)
    assert in_sys == out_sys == "wgs", "Only support `wgs84` coordnation system." # TODO
    print(f"Parse nodes: {timer.stop():.2f} s")

    # post-process
    timer.start()
    df_edges = pd.DataFrame(way_handler.edges)
    _size = df_edges.shape[0]
    df_edges.query('way_id not in @drop_idxs', inplace=True)
    types = df_edges.waypoints.apply(lambda x: type(x)).unique()
    if len(types) == 1 and types[0] == str:
        df_edges.loc[:, 'waypoints'] = df_edges.loc[:, 'waypoints'].apply(eval)
    if df_edges.shape[0] != _size:
        print(f"\tDrop duplicates ways / edges: {_size} -> {df_edges.shape[0]}")

    df_edges.loc[:, 'dist'] = cal_od_straight_distance(df_edges, df_nodes)
    print(f"Transform to Dataframe: {timer.stop():.2f} s, node len: {df_nodes.shape[0]}")

    return df_ways, df_nodes, df_edges

def _transform_coords_seq_2_linstring(waypoints, df_nodes):
    # Execute in the as small as possible
    coords = waypoints.apply(
        lambda x: eval(x) if isinstance(x, str) else x)
    geoms = coords.apply(
        lambda x: LineString(points_geoseries_2_ndarray(df_nodes.loc[x].geometry)))
    
    return geoms

@timeit
def _parrallel_collect_geoms(df, df_nodes, n_jobs=16):
    _size = df.shape[0] // n_jobs + 1
    df.loc[:, 'part'] = df.index // _size

    params = ((df.waypoints, df_nodes) for _, df in df.groupby('part'))
    df.drop(columns=['part'], inplace=True)

    geoms = parallel_process(_transform_coords_seq_2_linstring, params, pbar_switch=True, 
                            n_jobs=n_jobs, total=n_jobs, desc='Collect coords')

    geoms = pd.concat(geoms).sort_index()

    return geoms

@timeit
def _simplify_edges(df_edges, df_nodes, n_jobs):
    ori_df_edges = df_edges.copy()
    signal_control_points = df_nodes[~df_nodes.ctrl_type.isna()].index.unique()
    _edges = pipeline_combine_links(ori_df_edges, signal_control_points, n_jobs)
    print(f"\tSimplify the graph, len {ori_df_edges.shape[0]} -> {_edges.shape[0]}, {(1 - _edges.shape[0] / ori_df_edges.shape[0]) * 100:.2f} % off")

    return _edges

@timeit
def _add_revert_edges(df_edges, df_ways):
    _size = df_edges.shape[0]
    df_edges = add_reverse_edge(df_edges, df_ways, offset=False)
    print(f"\tAdd revert edge, len {_size} -> {df_edges.shape[0]}, {(df_edges.shape[0] / _size - 1) * 100:.2f} % up")

    return df_edges

@timeit
def _process_multi_edges(df_edges, ori_df_edges, df_ways):
    if "level" not in list(df_edges):
        df_edges = df_edges.merge(df_ways[['level']], left_on='way_id', right_index=True, how='left')
    
    # (src, dst) 重复的记录
    idxs = check_multi_edges(df_edges)
    normal_edges = df_edges.query("index not in @idxs")
    multi_edges = df_edges.loc[idxs].sort_values(
            ['src', 'dst', 'dist', 'level', 'way_id'], ascending=[True, True, True, True, True])

    # case 1: waypoints 一致的情况
    multi_edges, keep_idxs_case1, case1_drop_idxs = drop_duplicates_way(multi_edges, att='waypoints')
    if case1_drop_idxs:
        df_edges.loc[keep_idxs_case1.union(case1_drop_idxs)]
        assert multi_edges.query(f"index == {list(case1_drop_idxs)[0]}").shape[0] == 0
        logger.debug(f"Drop same waypoints: {case1_drop_idxs}")

    # case 2: waypoints 内容不一致
    keep_idxs = multi_edges.groupby(['src', 'dst']).head(1).index
    keeped_multi_edges = multi_edges.loc[keep_idxs]
    case2_drop_idxs = set([i for i in idxs if i not in keep_idxs])
    case2_drop_idxs.difference_update(case1_drop_idxs)

    if case2_drop_idxs:
        od_set = df_edges.loc[case2_drop_idxs].waypoints.apply(__extract_waypoints_2_od).tolist()
        od_set = set(np.concatenate(od_set))
        df_edges_waypoints = ori_df_edges.waypoints.apply(str).to_frame()

        pos = df_edges_waypoints.query("waypoints in @od_set")
        pos_od_idxs = pos.index
        pos_od_set = set(pos.waypoints.values)
        pos_df_edges = ori_df_edges.loc[pos_od_idxs]
        pos_df_edges.loc[:, 'dir'] = 1

        neg_od_set = od_set - pos_od_set
        neg_od_set = {str(eval(x)[::-1]) for x in neg_od_set}
        neg_od_set = df_edges_waypoints.query("waypoints in @neg_od_set").index
        neg_df_edges = swap_od(ori_df_edges.loc[neg_od_set])
        neg_df_edges.loc[:, 'dir'] = -1

        _multi_edges = pd.concat([pos_df_edges, neg_df_edges])
        logger.debug(f"od list len: {len(od_set)}, df_segs len: {_multi_edges.shape[0]}")
    else:
        _multi_edges = None

    # case 3: 针对 waypoints 仅有两条记录的情况
    ps = multi_edges.loc[case2_drop_idxs].waypoints
    tmp = df_edges.loc[ps[ps.apply(len) == 2].index]
    assert tmp.shape[0] == 0

    df_edges = pd.concat([normal_edges, keeped_multi_edges, _multi_edges])\
                 .drop(columns='level').reset_index(drop=True)

    return df_edges

def parse_xml_to_graph(fn, highway_filters=highway_filters, simplify=True, twoway=True, offset=True, 
                       crs="epsg:4326", in_sys='wgs', out_sys='wgs', n_jobs=16):   

    df_ways, df_nodes, df_edges_raw = _parse_xml(fn, highway_filters, crs, in_sys, out_sys)

    ori_df_edges = df_edges_raw.copy()
    if simplify:
        df_edges = _simplify_edges(ori_df_edges, df_nodes, n_jobs=n_jobs)
    
    if twoway:
        df_edges = _add_revert_edges(df_edges, df_ways)

    df_edges = _process_multi_edges(df_edges, ori_df_edges, df_ways)

    # df_edges = _transform_coords_seq_2_linstring(df_edges, df_nodes)
    df_edges.loc[:, 'geometry'] = _parrallel_collect_geoms(df_edges[['waypoints']], df_nodes, n_jobs)
    df_edges = gpd.GeoDataFrame(df_edges, crs=crs)

    if offset:
        df_edges = edge_offset(df_edges).sort_index()

    way_attrs = ['name', 'level', 'highway', 'link', 'maxspeed', 'oneway', 'lanes', 'service']
    df_edges = append_way_info(df_edges, df_ways, way_attrs)

    df_edges.reset_index(inplace=True, drop=True)
    df_edges.loc[:, 'eid'] = df_edges.index.values

    multi_edge_idxs = check_multi_edges(df_edges)
    assert len(multi_edge_idxs) == 0, "Check multi-edges"

    return df_nodes, df_edges, df_ways


#%%
if __name__ == "__main__":
    highway_filters=highway_filters; simplify=True; twoway=True; offset=True; crs="epsg:4326"; in_sys='wgs'; out_sys='wgs'; n_jobs=16
    fn = "./data/network/Shenzhen.osm.xml"

    timer = Timer()
    df_nodes, df_edges, df_ways = parse_xml_to_graph(fn)

    """ 测试最短路算法 """
    timer.start()
    from graph.geograph import GeoDigraph
    digraph = GeoDigraph(df_edges, df_nodes)

    timer.start()
    path = digraph.search(src=7959990710, dst=499265789)

    print(f"Map mathcing: {timer.stop():.2f} s")


    """ _add_reverse_edge """
    df_edges.query('way_id == 160131332').plot()

