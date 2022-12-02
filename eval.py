import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from mapmatching import ST_Matching, build_geograph
from mapmatching.setting import DATA_FOLDER
from mapmatching.utils.timer import Timer

from loguru import logger

def save_lables(res, fn):
    with open(fn, 'w') as f:
        json.dump(res, f)


def load_labels(fn):
    with open(fn, 'r') as f:
        _dict = json.load(f)

    _dict = {k:np.array(v) for k, v in _dict.items() }

    return _dict


def evaluation(matcher, trajs_folder, debug_folder=None):

    trajs = trajs_folder.glob("*.geojson")
    gt_fn = trajs_folder / 'gt.json'
    labels = load_labels(gt_fn)

    if debug_folder is None:
        debug_folder = DATA_FOLDER / "result"
    debug_folder.mkdir(exist_ok = True)
    
    preds = {}
    hit = 0
    errors = {}
    timer = Timer()
    timer.start()
        
    for fn in tqdm(trajs):
        name = fn.name
        traj = matcher.load_points(fn, compress=False)
        path, i = matcher.matching(traj, plot=False, top_k=5, dir_trans=True, 
                                   debug_in_levels=False, save_fn = debug_folder / str(name).replace('geojson', 'jpg'))
        y = path.eid.values
        # preds[fn.name] = [int(i) for i in y]

        if np.array(y == labels[name]).all():
            hit += 1
        else:
            errors[name] = fn

    print(f"Prcision: {hit / (hit + len(errors)) * 100:.1f} %, time cost: {timer.stop():.2f} s")
    if len(errors):
        print(f"Errors: {errors.keys()}")
        
    return 


if __name__ == "__main__":
    trajs_folder = DATA_FOLDER / "trajs"

    net = build_geograph(ckpt = DATA_FOLDER / 'network/Shenzhen_graph_pygeos.ckpt')
    matcher = ST_Matching(net=net)

    evaluation(matcher, trajs_folder, debug_folder=Path("./debug"))
    
    # net.search_memo.keys()

    # import pandas as pd
    # pd.DataFrame.from_dict(net.search_memo, orient='index')
    # from mapmatching.utils.serialization import save_checkpoint

    # save_checkpoint(net.search_memo, './data/debug/astar_search_memo.pkl')

