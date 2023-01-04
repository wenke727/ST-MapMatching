import pandas as pd
from datetime import date

def get_date(fmt="%Y-%m-%d"):
    return date.today().strftime(fmt)


def add_datetime_attr(nodes):
    extract_date_from_pid = lambda pid: {'area': pid[: 10], "date": pid[10: 16], "time": pid[16: 22]}
    nodes.loc[:, ['area', 'date', 'time']] = nodes.apply(lambda x: extract_date_from_pid(x.pid), axis=1, result_type='expand')

    return nodes

def SET_PANDAS_LOG_FORMET():
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 5000)

    return

