from tqdm import tqdm
from multiprocessing import cpu_count, Pool


def parallel_process(func, queue, pbar_switch=False, desc='Parallel processing', total=None, n_jobs=-1):
    """parallel process helper

    Args:
        func (Function): The func need to be parallel accelerated.
        queue ([tuple, tuple, ..., tuple]): the columns in df must contains the parmas in the func.
        desc (str, optional): [description]. Defaults to 'Parallel processing'.
        n_jobs (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
    """
    size = total
    if hasattr(queue, "__len__"):
        size = len(queue)
        if size == 0:
            return []
    
    n_jobs = cpu_count() if n_jobs == -1 or n_jobs > cpu_count() else n_jobs
    pool = Pool(n_jobs)
    
    if pbar_switch:
        pbar = tqdm(total=size, desc=desc)
        update = lambda *args: pbar.update()

    res = []
    for id, params in enumerate(queue):
        tmp = pool.apply_async(func, params, callback=update if pbar_switch else None)
        res.append(tmp)
    pool.close()
    pool.join() 
    res = [r.get() for r in res]

    return res


def _add(x, y):    
    res = x + y 
    # print(f"{x} + {y} = {res}")
    
    return res


def parallel_process_for_df(df, pipeline, n_jobs=8):
    """_summary_

    Args:
        df (_type_): _description_
        pipeline (_type_): _description_
        n_jobs (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    # FIXME 多进程中多个参数的情况下，现有代码是串行的, 因此 pipeline 中需要固定其他的参数
    import pandas as pd
    
    _size = df.shape[0] // n_jobs + 1
    df.loc[:, 'part'] = df.index // _size
    params = zip(df.groupby('part'))
    df.drop(columns=['part'], inplace=True)

    res = parallel_process(pipeline, params, n_jobs=n_jobs)
    sorted(res, key=lambda x: x[0])

    return pd.concat([i for _, i in res])


def pipeline_for_df_test(df_tuple, bias=-2, verbose=True):
    import time 
    name, df = df_tuple
    if verbose:
        print(f"Part {name} start, size: {df.shape[0]}\n")

    time.sleep(10)
    res = df.x + df.y + bias
    if verbose: 
        print(f"Part {name} Done\n")
    
    return name, res    


if __name__ == "__main__":
    res = parallel_process(_add, ((i, i) for i in range(10000)), True)

    # 基于 DataFrame 的多进程版本示例
    import pandas as pd
    df = pd.DataFrame({'x': range(0, 10000), 'y': range(0, 10000)})
    ans = parallel_process_for_df(df, pipeline_for_df_test, n_jobs=8)
    
    ans 