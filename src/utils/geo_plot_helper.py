import sys
import matplotlib.pyplot as plt
import geopandas as gpd
from haversine import haversine
import math

sys.path.append('/home/pcl/traffic/map_factory')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

from ImageRelatedProcess import clip_background, merge_tiles
import GoogleMapTile_V3 as tile


def adaptive_zoom_level(max_, min_):
    
    return    

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

    zoom = 15 - int(math.log2(haversine((x0, y1), (x1, y0))/3))
    # print([x0, x1], [y0, y1], haversine((x0, y1), (x1, y0))/3)

    # warming: if zoom big than 19 then there will be somthing wrong
    zoom = 19 if zoom > 19 else zoom

    img = tile.Tiles()
    f_lst, img_bbox = img.get_tiles_by_bbox([x0, y1, x1, y0], zoom, lyrs)
    to_image        = merge_tiles(f_lst)
    background, _   = clip_background( to_image, img_bbox, [x0, y1, x1, y0], False)

    ax.imshow(background, extent=[x0, x1, y0, y1], alpha=.6, zorder=0)
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
    map_visualize(df)
    