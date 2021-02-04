import traceback
from sigma.fontstyles import colors
from sigma.utils import timestamp
import os.path

class Logger():
    def __init__(self, name, mode='w', log_time=True, flush=True):
        self.log_time= log_time
        try:
            self.__logger = open(name, mode=mode)
        except Exception as e:
            raise IOError('open logger {} failed.'.format(colors.red(name)))
        self.phase = 'T'
        self.suspend = False
        self.flush = flush

    def __del__(self):
        if not self.__logger.closed:
            self.__logger.flush()
            self.__logger.close()

    def log(self, message, epoch=None, iteration=None):
        if not self.suspend and message is not None:
            string = self.phase
            if self.log_time:
                string = '{} {}'.format(timestamp(), string)
            if epoch is not None:
                string = '{} {}'.format(string, epoch)
            if iteration is not None:
                string = '{} {}'.format(string, iteration)
            string = '{} {}'.format(string, message)
            self.__logger.write('{}\n'.format(string))
            if self.flush:
                self.__logger.flush()

    def toggle_state(self, state=None):
        if state is None:
            self.suspend = not self.suspend
        else:
            self.suspend = state

    def train(self):
        self.phase = 'T'

    def eval(self):
        self.phase = 'E'

    def close(self):
        self.__logger.close()


def save_config(path, args, mode='w'):
    with open(path, mode=mode) as config:
        if mode == 'a':
            config.write('>>>>>><<<<<<\n')
        config.write('============\n')
        config.write('keys | value\n')
        config.write('============\n')
        a = vars(args)
        for key, value in a.items():
            config.write('{} | {}\n'.format(key, value))

            
def load_config(path):
    config = {}
    with open(path, mode='r') as fp:
        lines = fp.readlines()
        for line in lines[3:]:
            line = line.strip('\n')
            key,value = line.split('|',1)
            config[key.strip(' ')] = value.strip(' ')
    return config


def save_configs(path, margs, args, mode='w'):
    with open(path, mode=mode) as config:
        if mode == 'a':
            config.write('>>>>>><<<<<<\n')
        config.write('============\n')
        config.write('keys | value\n')
        config.write('============\n')
        for argue in [margs, args]:
            a = vars(argue)
            for key, value in a.items():
                config.write('{} | {}\n'.format(key, value))
