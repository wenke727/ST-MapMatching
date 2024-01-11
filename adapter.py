#%%
from copy import deepcopy
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import LineString
from shapely.geometry import MultiLineString

from mapmatching.graph import GeoDigraph
from mapmatching import ST_Matching
from mapmatching.utils.logger_helper import logger_dataframe, make_logger
from mapmatching.geo.io import read_csv_to_geodataframe, to_geojson

logger = make_logger('./debug', console=True)

def test_shortest_path(net, src, dst):
    """ 最短路径测试 """
    res = net.search(src, dst)
    df_edges.loc[res['epath']].plot()

    return res

def process_path_data(df):
    def aggregate_geometries(geom_list):
        # lines = [loads(geom) for geom in geom_list if geom != 'LINESTRING EMPTY']
        return MultiLineString(geom_list) if geom_list else 'LINESTRING EMPTY'

    df = df.copy()
    special_cases_mask = df['dst_name'].isin(['exchange', 'inner_link'])
    
    # step id
    step_id = 0
    _len = len(df)
    arr_steps = np.zeros(_len)
    df.loc[:, 'order'] = range(_len)
    prev_way_id = df.iloc[0].way_id
    for i, (_, item) in enumerate(df.iloc[1:].iterrows(), start=1):
        if item['way_id'] == prev_way_id and not special_cases_mask.iloc[i]:
            arr_steps[i] = step_id
            continue
        step_id += 1
        arr_steps[i] = step_id
        prev_way_id = item['way_id']
    df.loc[:, 'step'] = arr_steps

    # Separate records where dst_name is 'exchange' or 'inner_link'
    special_cases = df[special_cases_mask]
    df = df[~special_cases_mask]
    
    # Group by eid and aggregate
    grouped = df.groupby(['way_id', 'step']).agg({
        'src': 'first',
        'dst': 'last',
        'src_name': 'first',
        'dst_name': 'last',
        'eid': lambda x: list(x),
        # 'dir': lambda x: list(x),
        'distance': 'sum',
        'duration': 'sum',
        'walking_duration': 'sum',
        'speed': 'mean',
        'geometry': list,
        'dist': 'sum',
        'order': 'first',
    }).reset_index()

    # Handle missing values in walking_duration
    grouped['walking_duration'] = grouped['walking_duration'].replace({0: np.nan})

    # Combine the grouped data with the special cases
    result = pd.concat([grouped, special_cases], ignore_index=True)\
               .sort_values(['order', 'step'])\
               .drop(columns=['step', 'order'])\
               .reset_index(drop=True)

    return result


df_nodes = gpd.read_file('../MapTools/exp/shezhen_subway_nodes.geojson')
df_edges = gpd.read_file('../MapTools/exp/shezhen_subway_edges.geojson')

df_edges = df_edges.assign(
    dist = df_edges['distance'],
    geometry = df_edges.geometry.fillna(LineString())
)

net = GeoDigraph(df_edges, df_nodes.set_index('nid'), weight='duration')
matcher = ST_Matching(net=net, ll=False, loc_deviaction=100)


# %%
# FIXME 4, 14, 420
id = 420
fn = Path(f'./data/cells/{id:03d}.csv')

traj = read_csv_to_geodataframe(fn)
idxs = range(len(traj))
if fn.name == '004.csv':
    idxs = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 17, 18, 24, 25, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 50, 51]
if fn.name == '014.csv':
    # BUG 最短路问题, 需要增加 7
    idxs = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 28]
if fn.name == '420.csv':
    idxs = [0, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30, 31, 32]
traj = traj.loc[idxs] 


res = matcher.matching(traj, top_k=8, dir_trans=False, details=True, plot=True, 
                       search_radius=800, simplify=True, debug_in_levels=False)

processed_data = process_path_data(df_edges.loc[res['epath']])


info = deepcopy(res['probs'])
info.update({'step_0': res['step_0'], 'step_n': res['step_n']})
print(pd.DataFrame([info]))

processed_data

# %%
cands = res['details']['cands']
graph = res['details']['graph']
steps = res['details']['steps']
steps



# %%
# test_shortest_path(net, 440300024064012, 440300024056016) # 1号线，车公庙 -> 后海
# test_shortest_path(net, 440300024056014, 440300024056016) # 11号线，车公庙 -> 后海
# test_shortest_path(net, 440300024064012, 440300024056014) # 1号线，车公庙 -> 11号线，车公庙

# %%
