
import io
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from typing import List
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from .candidatesGraph import get_shortest_geometry

from ..graph import GeoDigraph
from ..utils.img import merge_np_imgs
from ..utils.parallel_helper import parallel_process
from ..geo.vis import plot_geodata, TILEMAP_FLAG, add_basemap


def plot_matching_debug_pair(traj, edges, item, level, src, dst, ax=None, maximun=None, legend=True, factor=4):
    """
    Plot the matching situation of one pair of OD (origin-destination).

    Args:
        traj (pandas.core.frame.DataFrame): Trajectory data containing points.
        edges (geopandas.geodataframe.GeoDataFrame): Edge data.
        item (pandas.core.series.Series): One record in tList. The multi-index here is (src, dest).
        level (int): Level of the graph.
        src (str): Source point.
        dst (str): Destination point.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
        maximun (float, optional): Maximum value. Defaults to None.
        legend (bool, optional): Whether to show the legend. Defaults to True.
        factor (int, optional): Factor used for color determination. Defaults to 4.

    Returns:
        matplotlib.axes.Axes: Axes object.

    Example:
        plot_matching_debug_pair(traj_data, edges_data, graph_t.loc[1], level=2, src="A", dst="B")

    Notes:
        - The 'traj' DataFrame should contain trajectory data with required columns.
        - The 'edges' GeoDataFrame should contain edge data with required columns.
        - The 'item' Series represents one record in tList.
        - The 'level' parameter specifies the level of the graph.
        - The 'src' and 'dst' parameters represent the source and destination points.
        - The 'ax' parameter is an optional Axes object to plot on. If not provided, a new Axes object will be created.
        - The 'maximun' parameter is used for determining the maximum value. If provided, it helps determine the color of the plotted data.
        - The 'legend' parameter controls whether to show the legend.
        - The 'factor' parameter is a factor used for color determination.
        - The function returns the Axes object used for plotting.

    """
    points = traj.loc[[level, item.pid_1]]

    if ax is None:
        _, ax = plot_geodata(points, alpha=.6, color='white', reset_extent=False)
    else:
        traj.plot(ax=ax, alpha=.6, color='white')
        ax.axis('off')
        if TILEMAP_FLAG:
            add_basemap(ax=ax, alpha=.5, reset_extent=False)
        # plot_geodata(traj, scale=scale, alpha=.6, color='white', ax=ax)

    # OD
    point_0 = traj.loc[[level]]
    point_n = traj.loc[[item.pid_1]]
    point_0.plot(ax=ax, marker="D", label=f'O', zorder=8, facecolor="white", edgecolor='green')
    point_n.plot(ax=ax, marker="s", label=f'D', zorder=8, facecolor="white", edgecolor='blue')

    # OD label
    font_style = {'zorder': 9, "ha": "center", "va":"bottom"}
    trans_p_str = f"trans: {item.trans_prob:.2f}"
    if 'dir_prob' in item:
        trans_p_str += f" = {item.dist_prob:.2f} * {item.dir_prob:.2f}"
    ax.text(point_0.geometry.x, point_0.geometry.y, trans_p_str, color='green', **font_style)
    ax.text(point_n.geometry.x, point_n.geometry.y, f"oberv: {item.observ_prob:.2f}", color='blue',**font_style)

    # path
    edges.iloc[[0]].plot(ax=ax, linestyle='--', alpha=.7, label=f'{src}', color='green', linewidth=5)
    edges.iloc[[1]].plot(ax=ax, linestyle=':', alpha=.7, label=f'{dst}', color='blue', linewidth=5)
    gpd.GeoDataFrame(item).T.set_geometry('geometry').plot(ax=ax, color='red', label='Path', alpha=.6, linewidth=2)

    # aux
    prob = item.observ_prob * item.trans_prob
    info = f"{prob:.4f}"

    if maximun is not None and prob == maximun:
        color = 'red'
    elif maximun / prob < factor:
        color = 'blue'
    else:
        color = 'gray'
    ax.set_title(f"{src} -> {dst}: {info}", color = color)
    ax.set_axis_off()

    if legend: 
        ax.legend()
    
    return ax
    
def ops_matching_debug_subplot(traj, edges, item, layer_id, src, dst, maximun):
    ax = plot_matching_debug_pair(traj, edges, item, layer_id, src, dst, maximun=maximun)
    with io.BytesIO() as buffer:
        plt.savefig(buffer, bbox_inches='tight', pad_inches=0.1, dpi=300)
        image = Image.open(buffer)
        array = np.asarray(image)
    
    plt.close()

    return array

def plot_matching_debug_level(net, traj, graph, layer_id, n_jobs=32):
    df_layer = graph.loc[layer_id]
    if df_layer.empty:
        return None
    n_rows = df_layer.index.get_level_values(0).nunique()
    n_cols = df_layer.index.get_level_values(1).nunique()
    _max = (df_layer.observ_prob * df_layer.trans_prob).max()

    params = ((traj, net.get_edge([src, dst]), df_layer.loc[src].loc[dst], layer_id, src, dst, _max) 
                for src, dst in df_layer.index.values)
    img_arrs = parallel_process(ops_matching_debug_subplot, params, n_jobs=n_jobs, pbar_switch=True)
    img_np = merge_np_imgs(img_arrs, n_rows, n_cols)

    img = Image.fromarray(img_np).convert("RGB")

    return img

def debug_traj_matching(traj: gpd.GeoDataFrame, graph: gpd.GeoDataFrame, net: GeoDigraph,
                        level: List[int]=None, debug_folder: str='./debug'):
    """
    Perform matching debug for trajectory on multiple layers of the graph.

    Args:
        traj (gpd.GeoDataFrame): The trajectory GeoDataFrame.
        graph (gpd.GeoDataFrame): The graph GeoDataFrame.
        level (int or List[int], optional): The level(s) of the graph to perform matching debug. If None, all layers will be processed. Defaults to None.
        debug_folder (str, optional): The folder path to save debug images. Defaults to './debug'.

    Returns:
        PIL.Image.Image: The debug image of the last processed layer.

    Example:
        >>> trajectory = gpd.GeoDataFrame([...])  # Trajectory GeoDataFrame
        >>> graph = gpd.GeoDataFrame([...])  # Graph GeoDataFrame
        >>> matcher = Matcher()  # Matcher object
        >>> debug_image = matcher.matching_debug(trajectory, graph, level=2, debug_folder='./debug')
        >>> print(debug_image)

    Notes:
        - The 'traj' GeoDataFrame should contain the trajectory data with a 'geometry' column.
        - The 'graph' GeoDataFrame should contain the graph data with a 'geometry' column.
        - The 'level' parameter specifies the layer(s) of the graph to perform matching debug. It can be an integer or a list of integers.
        - If 'level' is None, the matching debug will be performed on all layers of the graph.
        - The 'debug_folder' parameter specifies the folder path to save the debug images.
        - The function will save debug images for each processed layer in the specified folder and return the debug image of the last processed layer.
    """
    if 'geometry' not in graph:
        graph = get_shortest_geometry(graph, 'geometry')
        graph = gpd.GeoDataFrame(graph, crs=net.utm_crs)
        graph = graph.set_geometry('geometry')

    traj = traj.to_crs(4326)
    graph = graph.to_crs(4326)
    
    if level is None:
        layer_ids = graph.index.get_level_values(0).unique().sort_values().values
    else:
        layer_ids = level if isinstance(level, list) else [level]

    for idx in layer_ids:
        img = plot_matching_debug_level(net, traj, graph, idx)
        img.save(os.path.join(debug_folder, f"level_{idx:02d}.jpg"))
    
    return img

def plot_matching_result(traj_points: gpd.GeoDataFrame, path: gpd.GeoDataFrame, net: GeoDigraph, 
                         info: dict = None, column=None, categorical=True, legend=True):
    traj_crs = traj_points.crs.to_epsg()
    if path is not None and traj_crs != path.crs.to_epsg():
        path = path.to_crs(traj_crs)
        _df = traj_points
    else:
        _df = gpd.GeoDataFrame(pd.concat([traj_points, path]))

    fig, ax = plot_geodata(_df, figsize=(18, 12), tile_alpha=.7, reset_extent=False, alpha=0)

    traj_points.plot(ax=ax, label='Trajectory', zorder=2, alpha=.5, color='b')
    traj_points.iloc[[0]].plot(ax=ax, label='Source', zorder=4, marker="*", color='orange')
    if isinstance(path, gpd.GeoDataFrame) and not path.empty:
        path.plot(ax=ax, color='r', label='Path', zorder=3, linewidth=2, alpha=.6)

    ax = net.add_edge_map(ax, traj_crs, color='black', label='roads', alpha=.3, zorder=2, linewidth=1)

    # append information
    if info:
        for att in ['epath', "step_0", "step_n", 'details']:
            if att not in info:
                continue
            info.pop(att)

        text = []
        if "probs" in info:
            probs = info.pop('probs')
            info.update(probs)
        
        for key, val in info.items():
            if 'prob' in key:
                _str = f"{key}: {val * 100: .1f} %"
            else:
                if isinstance(val, float):
                    _str = f"{key}: {val: .0f}"
                else:
                    _str = f"{key}: {val}"
            text.append(_str)

        x0, x1, y0, y1 = ax.axis()
        ax.text(x0 + (x1- x0)/50, y0 + (y1 - y0)/50, "\n".join(text),
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    if legend:
        ax.legend(loc='best')

    return fig, ax

# deprecated
def debug_gt_level(net, traj, df_layer, layer_id, n_jobs=16, debug_folder='./'):
    """PLot the matchings between levels (i, i+1)

    Args:
        traj ([type]): [description]
        tList ([type]): The candidate points.
        graph_t ([type]): [description]
        level ([type]): [description]
        net ([type]): [description]
        debug (bool, optional): [description]. Save or not

    Returns:
        [type]: [description]
    """

    rows = df_layer.index.get_level_values(0).unique()
    cols = df_layer.index.get_level_values(1).unique()
    n_rows, n_cols = len(rows), len(cols)

    _max = (df_layer.observ_prob * df_layer.trans_prob).max()
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    for i, src in enumerate(rows):
        for j, dst in enumerate(cols):
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1) 
            plot_matching_debug_pair(net, traj, df_layer.loc[src].loc[dst], layer_id, src, dst, ax=ax, maximun=_max)

    if 'dir_prob' in list(df_layer):
        _title = f'Level: {layer_id} [observ * trans (dis, dir)]'
    else:
        _title = f'Level: {layer_id} [observ * trans]'

    plt.suptitle(_title)
    plt.tight_layout()
    
    if debug_folder:
        t = time.strftime("%Y%m%d_%H", time.localtime()) 
        plt.savefig( os.path.join(debug_folder, f"{t}_level_{layer_id}.jpg"), dpi=300)
        plt.close()
        
    return True

def plot_matching(net, traj, cands, route, save_fn=None, satellite=True, column=None, categorical=True):
    def _base_plot(df):
        if column is not None and column in traj.columns:
            ax = df.plot(alpha=.3, column=column, categorical=categorical, legend=True)
        else:
            ax = df.plot(alpha=.3, color='black')
        ax.axis('off')
        
        return ax
    
    # plot，trajectory point
    _df = gpd.GeoDataFrame(pd.concat([traj, route]))
    if satellite:
        try:
            _, ax = plot_geodata(_df, alpha=0, tile_alpha=.5, reset_extent=True)
            if column is not None:
                traj.plot(alpha=0, column=column, categorical=categorical, legend=True, ax=ax)
        except:
            ax = _base_plot(_df)       
    else:
        ax = _base_plot(_df)
        
    traj.plot(ax=ax, color='blue', alpha=.5, label= 'Trajectory')
    traj.head(1).plot(ax=ax, marker = '*', color='red', zorder=9, label= 'Start point')
    
    # network
    edge_lst = net.spatial_query(box(*traj.total_bounds))
    net.get_edge(edge_lst).plot(ax=ax, color='black', linewidth=.8, alpha=.4, label='Network' )
    
    # candidate
    net.get_edge(cands.eid.values).plot(
        ax=ax, label='Candidates', color='blue', linestyle='--', linewidth=.8, alpha=.5)
    
    # route
    if route is not None:
        route.plot(ax=ax, label='Path', color='red', alpha=.5)
    
    ax.axis('off')
    if column is None:
        plt.legend(loc=1)
    
    if save_fn is not None:
        plt.tight_layout()
        plt.savefig(save_fn, dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()
    
    return ax

def _base_plot(df, column=None, categorical=True):
    if column is not None and column in df.columns:
        ax = df.plot(alpha=.3, column=column, categorical=categorical, legend=True)
    else:
        ax = df.plot(alpha=.3, color='black')
    ax.axis('off')
    
    return ax

