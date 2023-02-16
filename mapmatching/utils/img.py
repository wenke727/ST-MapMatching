import numpy as np

def merge_np_imgs(arrays, n_row, n_col):
    """
    Merge a set of tiles into a single array.

    Parameters
    ---------
    tiles : list of mercantile.Tile objects
        The tiles to merge.
    arrays : list of numpy arrays
        The corresponding arrays (image pixels) of the tiles. This list
        has the same length and order as the `tiles` argument.

    Returns
    -------
    img : np.ndarray
        Merged arrays.
    extent : tuple
        Bounding box [west, south, east, north] of the returned image
        in long/lat.
    """
    # get indices starting at zero
    indices = []
    for r in range(n_row):
        for c in range(n_col):
            indices.append((r, c))

    # the shape of individual tile images
    h, w, d = arrays[0].shape
    h = max([i.shape[0] for i in arrays])
    w = max([i.shape[1] for i in arrays])

    # empty merged tiles array to be filled in
    img = np.ones((h * n_row, w * n_col, d), dtype=np.uint8) * 255

    for ind, arr in zip(indices, arrays):
        y, x = ind
        _h, _w, _ = arr.shape
        ori_x = x * w + (w - _w) // 2
        ori_y = y * h + (h - _h) // 2
        img[ori_y : ori_y + _h, ori_x : ori_x + _w, :] = arr

    return img

