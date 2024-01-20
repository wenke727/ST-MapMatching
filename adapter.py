#%%
from copy import deepcopy
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely import LineString
from shapely.geometry import MultiLineString
from shapely.ops import linemerge
from shapely.geometry import LineString, MultiLineString

from mapmatching.graph import GeoDigraph
from mapmatching import ST_Matching
from mapmatching.utils.logger_helper import logger_dataframe, make_logger
from mapmatching.geo.io import read_csv_to_geodataframe, to_geojson

logger = make_logger('./debug', console=True)

def _test_shortest_path(net, src, dst):
    """ 最短路径测试 """
    res = net.search(src, dst)
    df_edges.loc[res['epath']].plot()

    return res

def _merge_linestrings(linestrings, to_multilinestring=False):
    """
    Merges a list of LineString objects into a MultiLineString or a single LineString
    using Shapely's linemerge function.
    
    Args:
    - linestrings (list): A list of LineString objects.
    - to_multilinestring (bool): If True, force output to be a MultiLineString.

    Returns:
    - LineString/MultiLineString: The merged LineString or MultiLineString object.
    """

    # Filter out any 'LINESTRING EMPTY' or equivalent from the list
    valid_linestrings = [ls for ls in linestrings if not ls.is_empty]

    # If the input is empty or all linestrings are empty, return an empty LineString
    if not valid_linestrings:
        return LineString()

    # Use linemerge to combine the linestrings
    merged = linemerge(valid_linestrings)

    # If to_multilinestring is True and the merged result is not a MultiLineString, convert it
    if to_multilinestring and not isinstance(merged, MultiLineString):
        return MultiLineString([merged])
    
    return merged

def _process_path_data(df):
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
        'dist': 'sum',
        # 'distance': 'sum',
        'duration': 'sum',
        'walking_duration': 'sum',
        'speed': 'mean',
        'geometry': _merge_linestrings,
        'order': 'first',
    }).reset_index()

    # Handle missing values in walking_duration
    grouped['walking_duration'] = grouped['walking_duration'].replace({0: np.nan})

    # Combine the grouped data with the special cases
    result = pd.concat([grouped, special_cases], ignore_index=True)\
               .sort_values(['order', 'step'])\
               .drop(columns=['step', 'order'])\
               .reset_index(drop=True)

    return gpd.GeoDataFrame(result)


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
id = 4
fn = Path(f'./data/cells/{id:03d}.csv')

traj = read_csv_to_geodataframe(fn)
idxs = range(len(traj))
if fn.name == '004.csv':
    idxs = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 14, 17, 18, 24, 25, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 50, 51]
if fn.name == '014.csv':
    idxs = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 28]
if fn.name == '420.csv':
    idxs = [0, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30, 31, 32]
traj = traj.loc[idxs] 


res = matcher.matching(traj, top_k=6, dir_trans=False, details=True, plot=True, tolerance=500,
                       search_radius=500, simplify=True, debug_in_levels=False)

# 裁剪首尾段
df_path = df_edges.loc[res['epath']]
eps = 0.1
start = 0 if res['step_0'] < eps else 1
end = -1 if res['step_n'] < eps else len(res['epath'])
df_path = df_path.iloc[start: end]

df_combined_path = _process_path_data(df_path)


info = deepcopy(res['probs'])
info.update({'step_0': res['step_0'], 'step_n': res['step_n']})
print(pd.DataFrame([info]))

aoi = ['way_id', 'src', 'dst', 'src_name', 'dst_name', 'eid', 'dist', 'duration', 'speed']
df_combined_path[aoi]

# %%
cands = res['details']['cands']
graph = res['details']['graph']
steps = res['details']['steps']
steps

#%%
traj

# %%
