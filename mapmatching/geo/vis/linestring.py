import numpy as np


def plot_linestring_with_arrows(gdf_line, ax, color='red'):
    coord_arrs = gdf_line.geometry.apply(lambda x: np.array(x.coords))
    gdf_line.plot(ax=ax, color=color)

    for coords in coord_arrs:
        # refs: https://wizardforcel.gitbooks.io/matplotlib-user-guide/content/4.5.html
        mid = coords.shape[0] // 2
        ax.annotate('', xy=(coords[mid+1] + coords[mid]) / 2, xytext=coords[mid],
                    arrowprops=dict(arrowstyle="-|>", color=color),
                    zorder=9
        )

    return
