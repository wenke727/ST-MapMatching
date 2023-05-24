import numpy as np

from ..geo.metric import lcss, edr, erp
from ..geo.ops.resample import resample_polyline_seq_to_point_seq, resample_point_seq


def eval(traj, res=None, path=None, resample=5, eps=10, metric='lcss', g=None):
    """
    lcss 的 dp 数组 循环部分，使用numba 加速，这个环节可以降低 10% 的时间消耗（20 ms） 
    """
    # BUG 
    assert res is not None or path is not None
    assert metric in ['lcss', 'edr', 'erp']
    
    if path is None:
        path = self.transform_res_2_path(res)

    if traj.crs.to_epsg() != path.crs.to_epsg():
        traj = traj.to_crs(path.crs.to_epsg())

    if resample:
        _, path_coords_np = resample_polyline_seq_to_point_seq(path.geometry, step=resample,)
        _, traj_coords_np = resample_point_seq(traj.geometry, step=resample)
    else:
        path_coords_np = np.concatenate(path.geometry.apply(lambda x: x.coords[:]).values)
        traj_coords_np = np.concatenate(traj.geometry.apply(lambda x: x.coords[:]).values)
        
    eval_funs = {
        'lcss': [lcss, (traj_coords_np, path_coords_np, eps, self.ll)], 
        'edr': [edr, (traj_coords_np, path_coords_np, eps)], 
        'edp': [erp, (traj_coords_np, path_coords_np, g)]
    }
    _eval = eval_funs[metric]

    return _eval[0](*_eval[1])