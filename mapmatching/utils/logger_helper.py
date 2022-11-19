import os
import time

def make_logger(folder, level='DEBUG', mode='w', console=False):
    from loguru import logger

    if not console:
        logger.remove()
    
    logger.add(
        os.path.join(folder, f"pano_base_{time.strftime('%Y-%m-%d', time.localtime())}.log"), 
        enqueue=True,  
        backtrace=True, 
        diagnose=True,
        level=level,
        mode=mode
    )
    
    return logger
