import numpy as np
import matplotlib.pyplot as plt
import os
import os.path


def load(fold):
    data = []
    label = []
    for root, dirs, files in os.walk(fold):
        for f in files:
            datum = np.loadtxt(os.path.join(root, f))
            name = f.rsplit('-', 2)[1]
            data.append(datum)
            label.append(name)

    return (data, label)



def plot(fold):
    data, label = load(fold)
    colors = ['r', 'g', 'b', 'y']
    fig, axes = plt.subplots(2,2)
    for idx, (datum, name) in enumerate(zip(data, label)):
        x = np.arange(datum.shape[0])
        axes[0][0].plot(x, datum[:, 0], label=name)
        axes[0][1].plot(x, datum[:, 1], label=name)
        axes[1][0].plot(x, datum[:, 2], label=name)
        axes[1][1].plot(x, datum[:, 3], label=name)

    axes[0][0].set_ylabel('train-loss')
    axes[0][0].legend()
    axes[0][0].grid(True)
    axes[0][1].set_ylabel('train-acc')
    axes[0][1].legend()
    axes[0][1].grid(True)
    axes[1][0].set_ylabel('test-loss')
    axes[1][0].legend()
    axes[1][0].grid(True)
    axes[1][1].set_ylabel('test-acc')
    axes[1][1].legend()
    axes[1][1].grid(True)

    plt.show()


if __name__ == '__main__':
    plot('tmp')
