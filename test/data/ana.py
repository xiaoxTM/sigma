import matplotlib.pyplot as plt
import numpy as np

files = ['glorot_normal', 'glorot_uniform',
         'he_normal', 'he_uniform',
         'lecun_normal', 'lecun_uniform'
        ]
styles = ['r-.', 'g-+',
          'b-.', 'y-+',
          'k-.', 'c-+'
         ]

for f, style in zip(files, styles):
    data = np.loadtxt('debug/keras_{}'.format(f))
    #data = np.loadtxt('debug/sigma_{}'.format(f))
    #data = kdata - sdata
    #print(np.mean(data))
    index = np.arange(0, data.shape[0], step=10, dtype=np.int32)
    data = data[index]   
    plt.plot(range(data.shape[0]), data, style)

plt.legend(files)
plt.grid(True)
plt.show()
