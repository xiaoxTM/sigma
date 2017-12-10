import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1)

true = np.loadtxt('data/true.txt')
x = range(len(true))

ax[1].plot(x, true, label='true distr')

for filename in ['tanh', 'sigmoid', 'leaky_relu', 'elu', 'linear']:
    loss = np.loadtxt('data/{}-loss.txt'.format(filename))
    y = np.loadtxt('data/{}-y.txt'.format(filename))
    ax[0].plot(range(len(loss)), loss, label='{}-loss'.format(filename))
    ax[1].plot(range(len(y)), y, label='{}-distr'.format(filename))

ax[0].legend()
ax[1].legend()
plt.show()

