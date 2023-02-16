try:
    TILEMAP_FLAG = True
    from tilemap import plot_geodata, add_basemap
except:
    TILEMAP_FLAG = False
    def plot_geodata(data, *args, **kwargs):
        return data.plot(*args, **kwargs)
    def add_basemap(ax, *args, **kwargs):
        return ax
