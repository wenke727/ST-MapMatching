#%%
import geopandas as gpd
from pathlib import Path
from shapely import LineString
from mapmatching.graph import GeoDigraph
from mapmatching import ST_Matching
from mapmatching.geo.io import read_csv_to_geodataframe, to_geojson

# FIXME 4, 14, 420


def test_shortest_path(net):
    """ 最短路径测试 """
    res = net.search(440300024057010, 440300024074011)
    df_edges.loc[res['epath']].plot()

    return res

df_nodes = gpd.read_file('../MapTools/exp/shezhen_subway_nodes.geojson')
df_edges = gpd.read_file('../MapTools/exp/shezhen_subway_edges.geojson')

df_edges = df_edges.assign(
    dist = df_edges['distance'],
    geometry = df_edges.geometry.fillna(LineString())
)

net = GeoDigraph(df_edges, df_nodes.set_index('nid'), weight='duration')
matcher = ST_Matching(net=net, ll=False)


# %%
fn = Path('./data/cells/004.csv')
traj = read_csv_to_geodataframe(fn)
if fn.name == '004.csv':
    traj = traj.loc[[46, 47, 49, 50, 51]] # 44, 45, 

res = matcher.matching(traj, 
                       top_k=5, dir_trans=True, details=True, plot=True,
                       simplify=True, debug_in_levels=False)

#%%
df_edges.loc[res['epath']].to_csv('epath.csv')

# %%

cands = res['details']['cands']
graph = res['details']['graph']
# graph.to_csv("./graph.csv")
graph.loc[3]


# %%
traj = read_csv_to_geodataframe('./data/cells/004.csv')
to_geojson(traj, './data/cells/004.geojson')



# %%
# to_geojson(df_edges, '../MapTools/exp/shezhen_subway_edges_add_eid.geojson')

# %%
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import MultiLineString
import numpy as np
from loguru import logger

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
        'eid': lambda x: list(x),
        'src_name': 'first',
        'dst_name': 'last',
        'dir': lambda x: list(x),
        'distance': 'sum',
        'duration': 'sum',
        'walking_duration': 'sum',
        'speed': 'mean',
        # 'geometry': lambda x: MultiLineString(x),
        'dist': 'sum'
    }).reset_index()

    # Handle missing values in walking_duration
    grouped['walking_duration'] = grouped['walking_duration'].replace({0: np.nan})

    # Combine the grouped data with the special cases
    result = pd.concat([grouped, special_cases], ignore_index=True).sort_values('step')

    return result

# Apply the function to the data
processed_data = process_path_data(df_edges.loc[res['epath']])

# Display the first few rows of the processed data
processed_data.head()

# %%
