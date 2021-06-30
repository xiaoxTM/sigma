import sys
import time
import os

if __name__ == '__main__':
    print('begin',sys.argv)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    time.sleep(10)
    print('end',sys.argv)
