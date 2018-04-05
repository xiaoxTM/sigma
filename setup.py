"""
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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
          'json',
          'pydot'
      ],
      zip_safe=False)
