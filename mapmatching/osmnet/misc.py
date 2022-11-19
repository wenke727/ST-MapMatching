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
