import networkx as nx

class GeoGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)


    """ vis """
    def add_edge_map(self, ax, *arg, **kwargs):
        return
        
    """ property"""
    @property
    def crs(self):
        return self.df_edges.crs
    
    @property
    def epsg(self):
        return self.df_edges.crs.to_epsg()

