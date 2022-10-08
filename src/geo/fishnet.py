import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import mercantile as mt
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def create_bbox_fishnet(bbox, zoom, plot=False):
    fishnet = {}
    for i in mt.tiles(*bbox, [zoom]):
        bbox = [round(i, 6) for i in mt.bounds(i)]
        fishnet[i] = box(*bbox)

    fishnet = gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries(fishnet)}, 
        crs='epsg:4326'
    )

    if plot:
        from tilemap import plot_geodata
        fig, ax = plot_geodata(fishnet, facecolor='None', edgecolor='black')
        ax.set_title(f"level {zoom}, size {fishnet.shape[0]}")
        
    return fishnet


def create_shape_fishnet(geom, zooms=[11, 13]):
    """create tile fishnet by a shape. 

    Args:
        fn (function): _description_
        zooms (list, optional): Zoom intervals. Defaults to [11, 13].

    Returns:
        GeoDataFrame: GeoDataFrame, index with tile xyz, geometry with bounds.
    """
    if isinstance(geom, gpd.GeoDataFrame):
        total_bounds = list(geom.total_bounds)
        _gdf = geom
    elif isinstance(geom, BaseGeometry):
        total_bounds = list(geom.bounds())
        _gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs='epsg"4326')
    else:
        raise "Unsupport geometry type"
        
    directions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    zoom_min, zoom_max = zooms
    fishnet_levels = {}

    for z in range(zoom_min, zoom_max + 1):
        fishnet = create_bbox_fishnet(total_bounds, z)
        fishnet = gpd.sjoin(fishnet, _gdf, op='intersects')[['geometry']]
        fishnet_levels[z] = fishnet

    keeps = set(fishnet_levels[zoom_max].index)
    for z in range(zoom_max - 1, zoom_min - 1, -1):
        for idx in fishnet_levels[z].index:
            x, y, z = idx
            _z = z + 1
            lst = []
            for dx, dy in directions:
                _x, _y = 2 * x + dx, 2 * y + dy
                node = (_x, _y, _z)
                if node in keeps:
                    lst.append(node)
            
            if len(lst) == 4:
                for i in lst:
                    keeps.remove(i)
                keeps.add(idx)
            
    fishnet = gpd.GeoDataFrame(pd.concat(fishnet_levels.values())).loc[list(keeps)]
    
    return fishnet

if __name__ == "__main__":
    geom_fn = '../../input/GBA_boundry.geojson'
    gba_bound = gpd.read_file(geom_fn)
    # gba_bound.loc[:, 'group'] = 'gba'
    # gba_bound = gba_bound.dissolve(by='group')

    fishnet = create_shape_fishnet(gba_bound)

