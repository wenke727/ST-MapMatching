
import os
import time
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from tilemap import plot_geodata, add_basemap


def matching_debug_subplot(net, traj, item, level, src, dst, ax=None, maximun=None, legend=True, scale=.9, factor=4):
    """Plot the matching situation of one pair of od.

    Args:
        item (pandas.core.series.Series): One record in tList. The multi-index here is (src, dest).
        net ([type], optional): [description]. Defaults to net.
        ax ([type], optional): [description]. Defaults to None.
        legend (bool, optional): [description]. Defaults to True.

    Returns:
        ax: Ax.
    
    Example:
        matching_debug_subplot(graph_t.loc[1])
    """
    if ax is None:
        _, ax = plot_geodata(traj, scale=scale, alpha=.6, color='white')
    else:
        traj.plot(ax=ax, alpha=.6, color='white')
        ax.axis('off')
        add_basemap(ax=ax, alpha=.5, reset_extent=False)
        # plot_geodata(traj, scale=scale, alpha=.6, color='white', ax=ax)

    # OD
    traj.loc[[level]].plot(ax=ax, marker="*", label=f'O ({src})', zorder=9)
    traj.loc[[item.pid_1]].plot(ax=ax, marker="s", label=f'D ({dst})', zorder=9)

    # path
    gpd.GeoDataFrame( item ).T.plot(ax=ax, color='red', label='path')
    net.get_edge([src]).plot(ax=ax, linestyle='--', alpha=.8, label=f'first({src})', color='green')
    net.get_edge([dst]).plot(ax=ax, linestyle=':', alpha=.8, label=f'last({dst})', color='black')

    # aux
    prob = item.observ_prob * item.f
    if 'f_dir' in item:
        info = f"{prob:.3f} = {item.observ_prob:.2f} * {item.f:.2f} ({item.v:.2f}, {item.f_dir:.2f})"
    else:
        info = f"{prob:.3f} = {item.observ_prob:.2f} * {item.f:.2f}"

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
    

def matching_debug_level(net, traj, df_layer, layer_id, debug_folder='./'):
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

    _max = (df_layer.observ_prob * df_layer.f).max()
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    for i, src in enumerate(rows):
        for j, dst in enumerate(cols):
            ax = plt.subplot(n_rows, n_cols, i * n_cols + j + 1) 
            matching_debug_subplot(net, traj, df_layer.loc[src].loc[dst], layer_id, src, dst, ax=ax, maximun=_max)

    plt.suptitle(f'Level: {layer_id} [observ / trans (dis, dir)]')
    plt.tight_layout()
    
    if debug_folder:
        t = time.strftime("%Y%m%d_%H", time.localtime()) 
        plt.savefig( os.path.join(debug_folder, f"{t}_level_{layer_id}.jpg"), dpi=300)
        plt.close()
        
    return True


def plot_matching(net, traj, cands, route, save_fn=None, satellite=True, column=None, categorical=True):
    def _base_plot():
        if column is not None and column in traj.columns:
            ax = traj.plot(alpha=.3, column=column, categorical=categorical, legend=True)
        else:
            ax = traj.plot(alpha=.3, color='black')
        ax.axis('off')
        
        return ax
    
    # plot，trajectory point
    if satellite:
        try:
            from tilemap import plot_geodata
            _, ax = plot_geodata(traj, alpha=.3, color='black', extra_imshow_args={'alpha':.5}, reset_extent=True)
            if column is not None:
                traj.plot(alpha=.3, column=column, categorical=categorical, legend=True, ax=ax)
        except:
            ax = _base_plot()       
    else:
        ax = _base_plot()
        
    traj.plot(ax=ax, color='blue', alpha=.5, label= 'Compressed')
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
        plt.legend(loc='best')
    
    if save_fn is not None:
        plt.tight_layout()
        plt.savefig(save_fn, dpi=300)
        # plt.close()
    
    return ax


if __name__ == "__main__":
    matching_debug_level(geograph, points, gt.loc[0], 0)

