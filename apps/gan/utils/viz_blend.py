import pickle
import gzip
from scipy import misc as sm

def load(filename):
    f = gzip.open(filename, 'rb')
    objs = pickle.load(f)
    f.close()
    return objs

def plot(filename, ax=None):
    # t-SNE embedding of the digits dataset
    x = load(filename)
    x, y = x[-2], x[-1]
    for idx, (sample, label) in enumerate(zip(x, y)):
        sm.imsave('tmp/{}-{}.png'.format(idx, int(label[0])), sample[:, :, 0])

if __name__ == '__main__':
    plot('exp/pickles/100/0.pkl')
