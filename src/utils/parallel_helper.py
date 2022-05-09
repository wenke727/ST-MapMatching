from tqdm import tqdm
from multiprocessing import cpu_count, Pool


def parallel_process(func, queue, pbar_switch=False, desc='Parallel processing', n_jobs=-1, logger=None):
    """parallel process helper

    Args:
        func (Function): The func need to be parallel accelerated.
        queue ([tuple, tuple, ..., tuple]): the columns in df must contains the parmas in the func.
        desc (str, optional): [description]. Defaults to 'Parallel processing'.
        n_jobs (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    if hasattr(queue, "__len__"):
        size = len(queue)
        if size == 0:
            return []
    
    n_jobs = cpu_count() if n_jobs == -1 or n_jobs > cpu_count() else n_jobs
    pool = Pool(n_jobs)
    
    if pbar_switch:
        pbar = tqdm(size, desc=desc)
        update = lambda *args: pbar.update()

    res = []
    for id, params in enumerate(queue):
        tmp = pool.apply_async(func, params, callback=update if pbar_switch else None)
        res.append(tmp)
    pool.close()
    pool.join() 
    res = [r.get() for r in res]

    return res


def add_(x, y):
    res = x + y 
    print(f"{x} + {y} = {res}")
    
    return res


if __name__ == "__main__":
    parallel_process(add_, [(i, i) for i in range(100)])

