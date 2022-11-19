import os
import sys
import time
import logbook
from logbook import Logger, TimedRotatingFileHandler
from logbook.more import ColorizedStderrHandler

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(FILE_DIR, '../../log/')
logbook.set_datetime_format('local')


def log_type(record, handler):
    log_info = "[{date}] [{level}] [{filename}] [{func_name}] [{lineno}] {msg}".format(
        date=record.time,                              # 日志时间
        level=record.level_name,                       # 日志等级
        filename=os.path.split(record.filename)[-1],   # 文件名
        func_name=record.func_name,                    # 函数名
        lineno=record.lineno,                          # 行号
        msg=record.message                             # 日志内容
    )
    
    return log_info


class LogHelper(object):
    def __init__(self, log_dir=BASE_DIR, log_name='log.log', backup_count=10, log_type=log_type, stdOutFlag=False):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            
        self.log_dir = log_dir
        self.backup_count = backup_count
        
        handler = TimedRotatingFileHandler(filename= os.path.join(self.log_dir, log_name),
                                        date_format='%Y-%m-%d',
                                        backup_count=self.backup_count)
        self.handler = handler
        if log_type is not None:
            handler.formatter = log_type
        handler.push_application()

        if not stdOutFlag:
            return
        
        handler_std = ColorizedStderrHandler(bubble=True)
        if log_type is not None:
            handler_std.formatter = log_type
        handler_std.push_application()

    def get_current_handler(self):
        return self.handler

    @staticmethod
    def make_logger(level, name=str(os.getpid())):
        return Logger(name=name, level=level)


def log_helper(log_file, content):
    log_file.write( f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, {content}\n" )
    return 


if __name__ == "__main__":
    g_log_helper = LogHelper(log_name='log.log', stdOutFlag=True)
    log = g_log_helper.make_logger(level=logbook.INFO)
    log.critical("critical")    # 严重错误，会导致程序退出
    log.error("error")          # 可控范围内的错误 
    log.warning("warning")      # 警告信息
    log.notice("notice")        # 大多情况下希望看到的记录
    log.info("info")            # 大多情况不希望看到的记录
    log.debug("debug")          # 调试程序时详细输出的记录
    pass
