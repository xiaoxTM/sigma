"""
    sigma, a deep neural network framework.
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

import os
import os.path

##############################################################################
def load_filename(listname,
                  num=None,
                  basepath=None,
                  sep=' ',
                  namefilter=None):
    """
        load filename from listname

        Attributes
        ----------
            listname : string
                       file name where to load filename
            num : int
                  number of filename to load. None means all filename
            sep : string
                  separator of filename and the corresponding ground truth filename
            namefilter : function
                  filters to filter out filename not need
                  namefilter need filename and, return True if accept, or False if want to filter out
    """
    fn = open(listname, 'r')
    lines = fn.readlines()
    fn.close()
    filelist = []
    gtlist = []
    if namefilter is None:
        namefilter = lambda x: True
    if num is not None:
       lines = lines[:num]
    if basepath is not None:
        if isinstance(basepath, str):
            filepath = basepath
            gtpath = basepath
        elif isinstance(basepath, (list, tuple)):
            if len(basepath) != 2:
                raise ValueError('base path must have length of 2. given {}'
                                 .format(len(basepath)))
            filepath = basepath[0]
            gtpath = basepath[1]
        else:
            raise TypeError('basepath can only be "None",'
                            '"list / tuple" and "str". given {}'
                            .format(type(basepath)))
    for line in lines:
        files = line.rstrip('\n\r')
        if sep in files:
            files = files.split(sep)
        if isinstance(files, str):
            if basepath is not None:
                files = os.path.join(filepath, files)
            if namefilter(files):
                filelist.append(files)
        elif isinstance(files, list):
            if len(files) == 2:
                if basepath is not None:
                    files[0] = os.path.join(filepath, files[0])
                    files[1] = os.path.join(gtpath, files[1])
                if namefilter(files[0]):
                    filelist.append(files[0])
                    gtlist.append(files[1])
            else:
                raise ValueError('files should have length of 2 for'
                                 ' [sample, ground truth]. given length {}'
                                 .format(len(files)))
        else:
            raise TypeError('files should be str or list. given {}'
                            .format(type(files)))
    if len(gtlist) == 0:
        gtlist = None
    return filelist, gtlist


##############################################################################
def load_filename_from_dir(imagedir,
                           gtdir=None,
                           gtext=None,
                           num=None,
                           namefilter=None):
    assert os.path.isdir(imagedir)
    assert gtdir is None or (os.path.isdir(gtdir) and gtext is not None)
    filelist = []
    gtlist   = []
    num_load = 0
    if namefilter is None:
        namefilter = lambda x: True
    if num is None:
        for root, dirs, files in os.walk(imagedir):
            for f in files:
                if namefilter(os.path.join(root, f)):
                    filelist.append(os.path.join(root, f))
                    if gtdir is not None:
                        gtlist.append(os.path.join(gtdir, f.rsplit('.', 1)[0]+'.'+gtext))
    else:
        if not isinstance(num, 'int'):
            raise TypeError('`num` must be int. given {}'.format(type(num)))
        if num <= 0:
            raise ValueError('`num` must be greater than zero. given {}'
                             .format(num))
        for root, dirs, files in os.walk(imagedir):
            for f in files:
                if num_load >= num:
                    break
                if namefilter(os.path.join(root, f)):
                    num_load += 1
                    filelist.append(os.path.join(root, f))
                    if gtdir is not None:
                        gtlist.append(os.path.join(gtdir, f.rsplit('.', 1)[0]+'.'+gtext))

    if len(gtlist) == 0:
        gtlist = None
    return filelist, gtlist
