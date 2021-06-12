import os
import os.path
import shutil
import glob

def backups(dst,files):
    assert isinstance(dst,str)
    assert isinstance(files,(str,list,tuple))
    if isinstance(files,str):
        files = [files]
    os.makedirs(dst,exist_ok=True)
    for f in files:
        real = os.path.realpath(f)
        if os.path.exists(real):
            dirname = os.path.dirname(f)
            if dirname is not None and dirname != '':
                if dirname[0] == '/':
                    dirname = dirname[1:]
                ddst = os.path.join(dst,dirname)
                os.makedirs(ddst,exist_ok=True)
            else:
                ddst = dst
            abspath = os.path.abspath(real)
            if os.path.isfile(abspath):
                shutil.copy2(abspath,ddst)
            elif os.path.isdir(abspath):
                base = os.path.basename(f)
                ddst = os.path.join(ddst,base)
                shutil.copytree(abspath,ddst)
            else:
                raise TypeError('File or Directory: {} is neither FILE nor DIRECTORY'.format(abspath))
        else:
            raise FileNotFoundError('File or Directory: {} not found'.format(real))

def wildcard_grab(pattern):
    return glob.glob(pattern)
