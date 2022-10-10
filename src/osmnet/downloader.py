
import numpy as np
from pathlib import Path


def download_osm_xml(fn, bbox, verbose=False):
    """Download OSM map of bbox from Internet.

    Args:
        fn (function): [description]
        bbox ([type]): [description]
        verbose (bool, optional): [description]. Defaults to False.
    """
    if type(fn) == str:
        fn = Path(fn)
        
    if fn.exists():
        return True

    fn.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Downloading {}".format(fn))
    
    if isinstance(bbox, list) or isinstance(bbox, np.array):
        bbox = ",".join(map(str, bbox))

    try:
        import requests
        # https://dev.overpass-api.de/overpass-doc/en/index.html
        # 通过参数控制的 API 可参考 https://github.com/categulario/map_matching/blob/master/mapmatching/overpass/streets.overpassql
        url = f'http://overpass-api.de/api/map?bbox={bbox}'
        
        print(f"url: {url}")
        r = requests.get(url, stream=True)
        with open(fn, 'wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)

        if verbose:
            print("Downloaded success.\n")

        return True
    except:
        return False
