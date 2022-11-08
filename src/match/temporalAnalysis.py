import pandas as pd
import numpy as np


def cos_similarity(self, path_, v_cal=30):
    # TODO cos similarity for speed
    # path_ = [5434742616, 7346193109, 7346193114, 5434742611, 7346193115, 5434742612, 7346193183, 7346193182]
    seg = [[path_[i-1], path_[i]] for i in range(1, len(path_))]
    v_roads = pd.DataFrame(seg, columns=['src', 'dst']).merge(self.edges,  on=['src', 'dst']).v.values
    
    num = np.sum(v_roads.T * v_cal)
    denom = np.linalg.norm(v_roads) * np.linalg.norm([v_cal for x in v_roads])
    cos = num / denom
    
    return cos

 