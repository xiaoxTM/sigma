from setuptools import setup

setup(name='sigma',
      version='0.1',
      description='A function-style deep learning framework built on top of TensorFlow',
      url='https://github.com/xiaoxTM/sigma.git',
      author='Renwu Gao',
      author_email='gilyou.public@gmail.com',
      license='GPL-3.0',
      packages=['sigma'],
      install_requires=[
          'numpy',
          'tensorflow',
          'h5py',
          'gzip',
          'pickle',
          'io',
          'multiprocessing',
          'scipy',
          'smtplib',
          'email',
          'subprocess',
          'functools',
          'json'
      ],
      zip_safe=False)
