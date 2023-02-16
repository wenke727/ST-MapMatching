import math
from . import plot_geodata

def plot_points_with_dir(points, heading=None, arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.6)):
    """plot points with dirs

    Args:
        points (_type_): _description_
        heading (_type_, optional): _description_. Defaults to None.
        arrowprops (_type_, optional): _description_. Defaults to dict(facecolor='blue', shrink=0.05, alpha=0.6).

    Returns:
        _type_: _description_

    Example:
        ```
        import shapely
        import geopandas as gpd
        from mapmatching.geo.vis.point import plot_point_with_dir

        gdf = gpd.GeoDataFrame({'geometry': [shapely.Point((113.912154, 22.784351))]})
        plot_points_with_dir(gdf, 347)
        ```
    """
    types = points.geom_type.unique()
    assert len(types) == 1 and types[0] == "Point", "check points geom_type"
    fig, ax = plot_geodata(points, zorder=2)

    if not heading:
        return  ax

    if isinstance(heading, (int, float)):
        heading = [heading] * points.shape[0]

    for i, geom in enumerate(points.geometry):
        x, y = geom.coords[0]
        x0, x1 = ax.get_xlim()
        aux_line_len = (x1-x0) / 12
        dy, dx = math.cos(heading[i]/180*math.pi) * aux_line_len, math.sin(heading[i]/180*math.pi) * aux_line_len
        ax.annotate('', xy=(x+dx, y+dy), xytext=(x,y), arrowprops=arrowprops, zorder=1)

    return ax

