#-*- coding:utf-8 -*-

import logging
import logging.handlers as lh
import time

__name_log = time.strftime('%Y_%m_%d', time.localtime())
__logfile = r'./log/' + __name_log + '.log'


def __getLogger__(dest=__logfile, level=logging.DEBUG):
    handler = lh.RotatingFileHandler(filename=dest, maxBytes=1024*1024)
    fmt = '<%(asctime)s> %(filename)s[line:%(lineno)s] message:"%(message)s" [%(levelname)s]'
    datefmt='%Y-%m-%d %H:%M:%S' 
    formatter = logging.Formatter(fmt, datefmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger('Alpha')
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger

__LOG__ = __getLogger__()

def getLogger():
    return __LOG__
