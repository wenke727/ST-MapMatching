import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from haversine import haversine, Unit


class Bunch(dict):
    """A dict with attribute-access"""

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __dir__(self):
        return self.keys()



def cal_od_straight_distance(df_edges, df_nodes, od=['src', 'dst']):
    dist = df_edges.merge(df_nodes[['x', 'y']], left_on=od[0], right_index=True, suffixes=('_0', '_1'))\
                   .merge(df_nodes[['x', 'y']], left_on=od[1], right_index=True, suffixes=('_0', '_1'))\
                   .apply(lambda x: haversine((x.y_0, x.x_0), (x.y_1, x.x_1), unit=Unit.METERS), axis=1)

    return dist

def points_2_polyline(df_nodes:gpd.GeoDataFrame, points:list):
    coords = []
    for p in points:
        item = df_nodes.loc[p]
        coords.append(item.geometry.coords[0])

    return LineString(coords)


def get_geom_length(geoms, from_crs='epsg:4326', to_crs='epsg:900913'):
    assert isinstance(geoms, (pd.Series, gpd.GeoSeries))
    
    if geoms.name != 'geometry':
        geoms.name = 'geometry'
    lines = gpd.GeoDataFrame(geoms, crs=from_crs)

    return lines.to_crs(to_crs).length


""" Deprecated """
def pipline_cal_polyline_dist(df_tuple, df_nodes, verbose=True):
    name, df = df_tuple
    if verbose:
        timer = Timer()
        timer.start()
        print(f"Part {name} start, size: {df.shape[0]}")

    geoms = df.apply(lambda x: points_2_polyline(df_nodes, [x.src, x.dst]), axis=1)
    dist = get_geom_length(geoms)
    
    res = pd.DataFrame({'geometry':geoms, 'dist':dist})
    
    if verbose: 
        print(f"Part {name} Done, {timer.stop():.2f} s")
    
    return name, res


def parallel_process_for_df(df, pipline, n_jobs=8):
    import pandas as pd
    
    _size = df.shape[0] // n_jobs + 1
    df.loc[:, 'part'] = df.index // _size
    params = zip(df.groupby('part'))
    df.drop(columns=['part'], inplace=True)

    res = parallel_process(pipline, params, n_jobs=n_jobs)
    sorted(res, key=lambda x: x[0])

    return pd.concat([i for _, i in res])
