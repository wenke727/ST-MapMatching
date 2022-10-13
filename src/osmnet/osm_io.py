import pickle


def load_graph(fn):
    with open(fn, 'rb') as f:
        graph = pickle.load(f)
    
    return graph['df_nodes'], graph['df_edges'], graph['df_ways']


def save_graph(df_nodes, df_edges, df_ways, fn):
    graph = {
        'df_nodes': df_nodes, 
        'df_edges': df_edges, 
        'df_ways': df_ways
    }

    with open(fn, 'wb') as f:
        pickle.dump(graph, f)
    
    return True
