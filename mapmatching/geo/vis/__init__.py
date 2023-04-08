try:
    TILEMAP_FLAG = True
    from tilemap import plot_geodata, add_basemap
except:
    TILEMAP_FLAG = False

    def plot_geodata(data, *args, **kwargs):
        return None, data.plot()
    
    def add_basemap(ax, *args, **kwargs):
        return ax
