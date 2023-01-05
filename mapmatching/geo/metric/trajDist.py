# refs: https://github.com/bguillouet/traj-dist

import numpy as np
from haversine import haversine_vector, Unit
from ..haversineDistance import haversine_matrix


def lcss(array1:np.ndarray, array2:np.ndarray, eps:float=10.0):
    """
    Usage
    -----
    The `Longuest-Common-Subsequence distance` (Spherical Geometry) between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0) x 2 numpy_array
    param t1 : len(t1) x 2 numpy_array
    eps : float
    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(array1)
    n1 = len(array2)
    
    dist_matrix = haversine_matrix(array1, array2, xy=True)
    M = dist_matrix.copy()
    mask = M < eps
    M[mask] = True
    M[~mask] = False
    
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if M[i - 1, j - 1]:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])

    val = float(C[n0][n1]) / min([n0, n1])

    return val


def edr(array1, array2, eps):
    """
    Usage
    -----
    The `Edit Distance on Real sequence` between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float
    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(array1)
    n1 = len(array2)
    
    dist_matrix = haversine_matrix(array1, array2, xy=True)
    M = dist_matrix.copy()
    mask = M < eps
    M[mask] = True
    M[~mask] = False
    M.astype(int)

    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            subcost = M[i -1, j - 1]
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    
    return edr


def erp(array1, array2, g):
    """
    Usage
    -----
    The `Edit distance with Real Penalty` between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """
    n0 = len(array1)
    n1 = len(array2)
    C = np.zeros((n0 + 1, n1 + 1))

    M = haversine_matrix(array1, array2, xy=True)

    ref_1 = haversine_vector(array1[:, ::-1], g[::-1], unit=Unit.METERS)
    ref_2 = haversine_vector(array2[:, ::-1], g[::-1], unit=Unit.METERS)

    C[1:, 0] = np.sum(ref_1)
    C[0, 1:] = np.sum(ref_2)
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + ref_1[i - 1]
            derp1 = C[i, j - 1] + ref_2[j - 1]
            derp01 = C[i - 1, j - 1] + M[i - 1, j - 1]
            C[i, j] = min(derp0, derp1, derp01)
    
    erp = C[n0, n1]
    
    return erp


""" Euclidean Geometry """
def e_lcss(t0, t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float
    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if eucl_dist(t0[i - 1], t1[j - 1]) < eps:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    lcss = 1 - float(C[n0][n1]) / min([n0, n1])
    return lcss


