#%%
import re
import osmium
import warnings
import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry import Point

warnings.filterwarnings('ignore')

import sys
sys.path.append('../')
from utils.timer import Timer
from misc import Bunch, cal_od_straight_distance
from osmnet.combine_edges import pipeline_combine_links


highway_filters = {'cycleway','footway','pedestrian','steps','track','corridor','elevator','escalator','service','living_street'}


class WayHandler(osmium.SimpleHandler):
    def __init__(self, link_type_filter=None):
        osmium.SimpleHandler.__init__(self)
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

        # ways
        way['osm_way_id'] = int(w.id)
        way['ref_node_id_list'] = [int(node.ref) for node in w.nodes]
        way['name'] = w.tags.get('name')
        self._extract_lane_info(w, way)
        self._extract_maxspeed_info(w, way)
        self._extract_one_way_info(w, way)
        self.osm_way_dict[way['osm_way_id']] = way

        self.way_nodes.update(way.ref_node_id_list)

        # edges
        nds = way.ref_node_id_list
        for i in range(len(way.ref_node_id_list)-1):
            self.edges.append({'way_id': way['osm_way_id'], 'order': i, 'src':nds[i], 'dst':nds[i+1]})        


    def _extract_one_way_info(self, w, way):
        oneway_info = w.tags.get('oneway')
        way['oneway'] = None
        if oneway_info is not None:
            if oneway_info == 'yes' or oneway_info == '1':
                way.oneway = True
            elif oneway_info == 'no' or oneway_info == '0':
                way.oneway = False
            elif oneway_info == '-1':
                way.oneway = True
                way.is_reversed = True
            elif oneway_info in ['reversible', 'alternating']:
                # todo: reversible, alternating: https://wiki.openstreetmap.org/wiki/Tag:oneway%3Dreversible
                way.oneway = False
            else:
                logger.warning(f'new lane type detected at way {way.osm_way_id}, {oneway_info}')   
        else:
            way.oneway = True


    def _extract_lane_info(self, w, way):
        lane_info = w.tags.get('lanes')
        way['lane_info'] = None
        if lane_info is not None:
            lanes = re.findall(r'\d+\.?\d*', lane_info)
            if len(lanes) > 0:
                way.lanes = int(float(lanes[0]))  # in case of decimals
            else:
                logger.warning(f'new lanes type detected at way {way.osm_way_id}, {lane_info}')
        lane_info = w.tags.get('lanes:forward')
        if lane_info is not None:
            try:
                way.forward_lanes = int(lane_info)
            except:
                pass
        lane_info = w.tags.get('lanes:backward')
        if lane_info is not None:
            try:
                way.backward_lanes = int(lane_info)
            except:
                pass        

    
    def _extract_maxspeed_info(self, w, way):
        maxspeed_info = w.tags.get('maxspeed')
        way['maxspeed'] = None
        if maxspeed_info is not None:
            try:
                way.maxspeed = int(float(maxspeed_info))
            except ValueError:
                speeds = re.findall(r'\d+\.?\d* mph', maxspeed_info)
                if len(speeds) > 0:
                    way.maxspeed = int(float(speeds[0][:-4]) * 1.6)
                else:
                    speeds = re.findall(r'\d+\.?\d* km/h', maxspeed_info)
                    if len(speeds) > 0:
                        way.maxspeed = int(float(speeds[0][:-5]))
                    else:
                        logger.warning(f'new maxspeed type detected at way {way.osm_way_id}, {maxspeed_info}')                            


class NodeHandler(osmium.SimpleHandler):
    def __init__(self, node_filter=None):
        osmium.SimpleHandler.__init__(self)
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

        # node = OSMNode(osm_node_name, osm_node_id, node_geometry, in_region, osm_highway, ctrl_type)
        item = {'name': osm_node_name, 
                'highway': osm_highway, 
                'ctrl_type': ctrl_type, 
                "x": lon,
                "y": lat,
                'geometry': node_geometry
        }
        
        self.nodes[osm_node_id] = item


def parse_xml_to_graph(fn, highway_filters=None, simplify=True, n_jobs=8):
    # ways
    timer = Timer()
    way_handler = WayHandler(highway_filters)
    way_handler.apply_file(fn)
    print(f"Parse ways: {timer.stop():.2f} s")
    
    # nodes
    timer.start()
    node_handler = NodeHandler(way_handler.way_nodes)
    node_handler.apply_file(fn)
    print(f"Parse nodes: {timer.stop():.2f} s")
    
    # post-process
    timer.start()
    df_nodes = gpd.GeoDataFrame.from_dict(node_handler.nodes, orient='index')
    df_nodes.set_crs('epsg:4326', inplace=True)

    df_edges = pd.DataFrame(way_handler.edges)
    df_edges.loc[:, 'dist'] = cal_od_straight_distance(df_edges, df_nodes)

    df_ways = pd.DataFrame.from_dict(way_handler.osm_way_dict, orient='index')
    df_ways.loc[:, 'src'] = df_ways.ref_node_id_list.apply(lambda x: x[0])
    df_ways.loc[:, 'dst'] = df_ways.ref_node_id_list.apply(lambda x: x[-1])
    print(f"Transform to Dataframe: {timer.stop():.2f} s, node len: {df_nodes.shape[0]}")

    # combine edges
    if simplify:
        ori_size = df_edges.shape[0]
        signal_control_points = df_nodes[~df_nodes.ctrl_type.isna()].index.unique()
        df_edges = pipeline_combine_links(df_edges, signal_control_points, n_jobs)
        print(f"Simplify the graph, {(1- df_edges.shape[0] / ori_size) * 100:.2f} % off")
      
    return df_nodes, df_edges, df_ways


#%%
if __name__ == "__main__":
    # fn = "/home/pcl/factory/ST-MapMatching/cache/GBA.osm.xml"
    fn = "/home/pcl/factory/ST-MapMatching/cache/Shenzhen.osm.xml"
    
    df_nodes, df_edges, df_ways = parse_xml_to_graph(fn)
            
    timer = Timer()
    # 测试最短路算法
    timer.start()
    from utils.DataStructure.digraph import DigraphAstar
    digraph = DigraphAstar(df_edges[['src', 'dst', 'dist']].values, df_nodes.to_dict(orient='index'))

    path = digraph.a_star(src=7959990710, dst=499265789)
    df_nodes.loc[path['path']].plot()

    print(path['path'])
    print(f"Map mathcing: {timer.stop():.2f} s")


# %%
