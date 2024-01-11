import os
import time
from loguru import logger
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000) # 打印结果不换行方法
pd.set_option('display.max_colwidth', 200)


def make_logger(folder, filename=None, level='DEBUG', mode='w', console=True, 
                rotation="10 MB", compression="zip", include_timestamp=True):
    """
    Creates a logger with specified settings.

    Args:
    - folder (str): Folder where the log file will be stored.
    - filename (str, optional): Name of the log file. Defaults to a date-based name.
    - level (str, optional): Logging level. Defaults to 'DEBUG'.
    - mode (str, optional): File mode. Defaults to 'w'.
    - console (bool, optional): If True, also log to console. Defaults to True.
    - rotation (str/int, optional): Rotate the log file at a certain interval or file size. Defaults to "10 MB".
    - compression (str, optional): Compression for rotated logs. Defaults to "zip".
    - include_timestamp (bool, optional): Include timestamp in the log file name. Defaults to True.

    Returns:
    - loguru.Logger: Configured logger object.
    """
    
    if not console:
        logger.remove()

    # Determine the log filename
    if filename is None:
        filename = "log"
    if include_timestamp:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        filename = f"{filename}_{timestamp}"
    log_filename = f"{filename}.log"
    log_path = os.path.join(folder, log_filename)

    try:
        logger.add(log_path, enqueue=True, backtrace=True, diagnose=True, 
                   level=level, mode=mode, rotation=rotation, 
                   compression=compression)
    except Exception as e:
        print(f"Error configuring logger: {e}")
        return None
    
    return logger

def logger_dataframe(df, desc="", level='debug'):
    getattr(logger, level)(f"{desc}\n{df}")

