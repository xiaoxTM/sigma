import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma.helpers import funs

@funs.stampit({'path':0, 'another':3})
def print_it(path, another, third):
    print('path:', 0)
    print('another:', -1)
    print(path)
    print(another)
    print(third)




if __name__ == '__main__':
    print_it(None, True, '/path/to/d')
