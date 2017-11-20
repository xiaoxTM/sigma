import argparse
import sys
import importlib
sys.path.append('/home/xiaox/studio/src/git-series')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str, default='ngan', help='framework model to use')
    parser.add_argument('-i', '--ninput', type=int, default=38, help='dimensions of input for noise')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='dimensions of input for noise')
    parser.add_argument('-s', '--shape', type=str, default='28x28x1', help='the real data shape')
    parser.add_argument('-t', '--train', type=bool, default=True, help='train or valid')
    args = parser.parse_args()
    shape = [int(x) for x in args.shape.split('x')]
    print('configuration for train `{}`'.format(args.network))
    print('\tsize of noise: {}'.format(args.ninput))
    print('\tshape of real data: {}'.format(shape))
    print('\ttrain: {}'.format(args.train))
    print('\tbatch-size: {}'.format(args.batch_size))
    module = importlib.import_module('models.{}'.format(args.network))
    module.run(args.ninput, shape, args.train, args.batch_size)
