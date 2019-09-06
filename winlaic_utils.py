import numpy as np
import os
import shutil
import time
import logging
import sys
import tqdm
from os.path import join


listt = lambda x: list(map(list, zip(*x)))

# 与 TQDM 配合的 Handler ，防止干扰进度条的打印。            
class TqdmLoggingHandler(logging.Handler):
    def __init__(self,level = logging.NOTSET):
        super().__init__(level)
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def clear(self):
        self.__init__()

    def add(self,value):
        self.sum += value
        self.count += 1

    @property
    def mean(self):
        return self.sum/self.count

class WinlaicLogger:
    '''自定义Logger：
    使用方法：指定 Logger 名称和存储目录，建立 Logger实例
    调用实例时直接对属性赋值，传入Log列表，格式同Matlab。
    如：
    lgr = Logger('winlaic')
    lgr.i=["accuracy",0.6,"loss",3.7]
    '''
    def __init__(self, logger_name='logs', loggers_directory='.'):
        logging_time = time.localtime()
        logging_directory = loggers_directory + os.sep + logger_name
        file_name = '%0.4d-%0.2d-%0.2d_%0.2d-%0.2d-%0.2d.log' % tuple(
            list(logging_time)[0:6])
        file_path = logging_directory + os.sep + file_name
        if not os.path.exists(logging_directory):
            os.makedirs(logging_directory)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        self.stream_handler = TqdmLoggingHandler()
        self.stream_handler.setLevel(logging.INFO)
        self.file_handler = logging.FileHandler(file_path)
        self.file_handler.setLevel(logging.DEBUG)

        self.formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s][%(message)s]')

        self.stream_handler.formatter = self.formatter
        self.file_handler.formatter = self.formatter

        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

    def parse_message_list(self, message_list):
        ret = ''
        for i_item, item in enumerate(message_list):
            ret += str(item)
            if i_item % 2 == 0:
                ret += ': '
            else:
                ret += '\t'
        return ret[0:-1]

    @property
    def d(self): pass

    @property
    def i(self): pass

    @property
    def w(self): pass

    @property
    def e(self): pass

    @w.setter
    def w(self, message_list):
        self.logger.warning(self.parse_message_list(message_list))

    @i.setter
    def i(self, message_list):
        self.logger.info(self.parse_message_list(message_list))

    @d.setter
    def d(self, message_list):
        self.logger.debug(self.parse_message_list(message_list))

    @e.setter
    def e(self, message_list):
        self.logger.error(self.parse_message_list(message_list))

def removeall(dir):
    if not os.path.exists(dir):return
    for file in os.listdir(dir):
        file = os.path.join(dir,file)
        if os.path.isdir(file):
            removeall(file)
        else:
            os.remove(file)
    shutil.rmtree(dir)
    
class ModelSaver():
    def __init__(self, dir='saved_models'):
        self.begin_time = time.localtime()
        self.dir = join('.', dir, '%0.4d-%0.2d-%0.2d_%0.2d-%0.2d-%0.2d' % tuple(list(self.begin_time)[0:6]))
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
    def save_dir(self, comments):
        assert isinstance(comments, list) and len(comments)%2 == 0, 'Wrong comment format.'
        saved_dir = '%0.4d-%0.2d-%0.2d_%0.2d-%0.2d-%0.2d' % tuple(list(time.localtime())[0:6])
        for i in range(len(comments)//2):
            saved_dir += '_{0}:{1}'.format(comments[i*2], comments[i*2+1])
        saved_dir = join(self.dir,saved_dir)
        os.makedirs(saved_dir)
        return saved_dir

