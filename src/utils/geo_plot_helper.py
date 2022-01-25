#%%
import sys
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from haversine import haversine, Unit

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# sys.path.append('/home/pcl/traffic/map_factory')
# from ImageRelatedProcess import clip_background, merge_tiles
# import GoogleMapTile_V3 as tile

 
def calculate_zoom(w, s, e, n, min_=12, max_=18):
    # Note: we adopt the caculate zoom logic from contextily (https://contextily.readthedocs.io/en/latest/index.html)
    """Automatically choose a zoom level given a desired number of tiles.

    .. note:: all values are interpreted as latitude / longitutde.

    Parameters
    ----------
    w : float
        The western bbox edge.
    s : float
        The southern bbox edge.
    e : float
        The eastern bbox edge.
    n : float
        The northern bbox edge.

    Returns
    -------
    zoom : int
        The zoom level to use in order to download this number of tiles.
    """
    # Calculate bounds of the bbox
    lon_range = np.sort([e, w])[::-1]
    lat_range = np.sort([s, n])[::-1]

    lon_length = np.subtract(*lon_range)
    lat_length = np.subtract(*lat_range)

    # Calculate the zoom
    zoom_lon = np.ceil(np.log2(360 * 2.0 / lon_length))
    zoom_lat = np.ceil(np.log2(360 * 2.0 / lat_length))
    zoom = int(np.max([zoom_lon, zoom_lat]))
    
    if zoom > max_:
        return max_
    if zoom < min_:
        return min_
    
    return zoom


def adaptive_zoom_level(w, s, e, n, ax, factor=10, max_=19, min_=10 ):
    dis = haversine((s, w), (n, e), unit=Unit.METERS)
    scale_dict = { i: 40076000/256 / np.power(2, i) for i in range(0, 23)}

    x, y = ax.get_figure().get_size_inches()
    diagonal_pixels = np.sqrt(x*x + y*y ) * ax.get_figure().get_dpi()
    act_scale = dis * factor / diagonal_pixels

    z = 0
    for key, val in scale_dict.items():
        if act_scale < val:
            continue
        z = key
        break
    z = z- 1

    if z <= min_:
        return min_
    if z >= max_:
        return max_
    
    return z


def map_visualize(df: gpd.GeoDataFrame, 
                  lyrs='s', 
                  scale=0.5, 
                  figsize = (12,9), 
                  color = "red", 
                  ax = None, 
                  fig=None, 
                  *args, **kwargs):
    """Draw the geodataframe with the satellite image as the background

    Args:
        `df` (gpd.GeoDataFrame): the gpd.GeoDataFrame need to plot
        `ax`: the ax define to draw
        `lyrs` (str, optional): [ m 路线图; t 地形图; p 带标签的地形图; s 卫星图; y 带标签的卫星图; h 标签层（路名、地名等）]. Defaults to 'p'.
        `scale` (float): border percentage
        `color`: the color the the geometry drawed

    Returns:
        [ax]: [description]
    """
    # lyrs='y';scale=0.5;figsize = (12,9); color = "red";ax = None;fig=None;
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    df.plot(color = color, ax=ax, zorder=1, *args, **kwargs)
    # df.plot(color = color, zorder=1)

    [x0, x1], [y0, y1] = plt.xlim(), plt.ylim()
    gap_x, gap_y = (x1-x0), (y1-y0)
    [a, b, c, d] = df.total_bounds

    if a == c:
        x0, x1 = a - 0.001, c + 0.001
        gap_x = x1- x0
    
    if b == d:
        y0, y1 = b - 0.001, d + 0.001
        gap_y = y1 - y0
        
    if not 0.4 <= gap_y / gap_x <= 2.5:
        mid_x, mid_y = (x1+x0)/2, (y1+y0)/2
        gap = max(gap_x, gap_y) * (1 + scale) / 2
        [x0, y0, x1, y1] = [mid_x - gap, mid_y - gap, mid_x + gap, mid_y + gap]
    else:
        [x0, y0, x1, y1] = [x0-(x1-x0) * scale, y0+(y0-y1) * scale,
                            x1+(x1-x0) * scale, y1-(y0-y1) * scale]

    zoom = calculate_zoom(x0, y0, x1, y1)
    # zoom = adaptive_zoom_level(x0, y0, x1, y1, ax)

    # TODO
    # img = tile.Tiles()
    # f_lst, img_bbox = img.get_tiles_by_bbox([x0, y1, x1, y0], zoom, lyrs)
    # to_image        = merge_tiles(f_lst)
    # background, _   = clip_background( to_image, img_bbox, [x0, y1, x1, y0], False)
    # ax.imshow(background, extent=[x0, x1, y0, y1], alpha=.6, zorder=0)

    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    
    # 去除科学记数法
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

    # set_major_locator
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    
    return fig, ax



if __name__ == '__main__':
    from shapely.geometry import LineString
    line = LineString([(113.932686, 22.583023), (113.932679, 22.583161), (113.932679, 22.583221), (113.932654, 22.583295)])
    df = gpd.GeoDataFrame( [{"geometry": line}] )
    _, ax = map_visualize(df)
    
    # daptive_zoom_level(*line.bounds, ax)
    adaptive_zoom_level(*line.bounds, ax)
    calculate_zoom(*line.bounds)
