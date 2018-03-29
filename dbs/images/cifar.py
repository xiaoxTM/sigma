import pickle

def load(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp, encoding='bytes')
    return data
