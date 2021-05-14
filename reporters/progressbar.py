import numpy as np
import sigma
from sigma.fontstyles import colors
from sigma.utils import intsize
from collections import Iterable,Iterator,Generator

class ProgressBar():
    def __init__(self,
                 items=None,
                 total:int=None,
                 num_prompts:int=20,
                 silent:bool=False,
                 parent=None,
                 keep_line:bool=True,
                 spec:str=None,
                 nc:str='x+',
                 cc:str='+')->None:
        ''' the value of `keep_line=False` from parent will conver the `keep_line` value of
            ProgressBar instance created by ProgressBar.sub()
            i.e.,
            p = ProgressBar()
            q = p.sub(keep_line=True)
            r = q.sub(keep_line=False)
            then, p._keep_line=False will not keep the print of q&r iteration in line.
            The only exception is that the keep line value of the last sub progressbar
            will not be effected by the its parents but its own states, i.e., if it is
            in progress, no matter all of the keep_line values are, line is keeped.
        '''
        self.__parent = parent
        if items is None:
            assert total is not None and total > 0,'`items` and `total` cannot be both None'
            self.__total = total
            self.__items = range(total)
        else:
            if isinstance(items,int):
                self.__total = items
                self.__items = range(items)
            elif isinstance(items,Iterable):
                self.__items = items
                self.__total = len(items)
            else:
                raise TypeError('`items` must be int/Iterable/Iterator/Generator, given {}'.format(type(items)))
        self.__iterator = None
        self.__index = -1
        self.__scale = float(num_prompts) / self.__total
        self._keep_line = keep_line
        #if parent is None:
        self.__nc = [c.encode('utf8') if isinstance(c,str) else c for c in nc]
        self.__cc = [c.encode('utf8') if isinstance(c,str) else c for c in cc]
        self.__prompts = np.asarray([self.__cc[0]] * num_prompts,dtype=np.string_)
        if parent is None:
            self.__total_size =sigma.utils.intsize(self.__total)
            self.__index_nc = 0
            self.__index_cc = 0
            self._spec = spec
            if self._spec is None:
                self._spec = '\r<{{:0>{}}}{}{}> [{{:{}}}, {{:3}}%] {{}}' \
                             .format(self.__total_size,
                                     colors.green('@'),
                                     colors.blue(str(self.__total)),
                                     num_prompts)
            def format_message(index,scale,percentage,message=None):
                idx = max(int(scale * (index+1)),int(scale * index)+1)
                if message is None:
                    message = ''
                return self._spec.format(self.__index+1,self.__prompts[:idx].tostring().decode('utf-8'),percentage,message)
            self._format_message = format_message

        self._iter_print_message = not silent
        self._silent = silent
        self._message = None

    def set_message(self,message):
        self._message = message

    def reset(self):
        if self.__parent is not None:
            self.__parent.reset()
        else:
            self.__prompts[:] = self.__cc[0]
        self.__index = -1
        self.__index_nc = 0
        self.__index_cc = 0

    @property
    def prompts(self):
        return self.__prompts

    def sub(self,items,total:int=None,keep_line:bool=True,nc:str=None,cc:str=None):
        self._iter_print_message = False
        if self.__parent is None: # itself is parent
            num_prompts = self.__prompts.shape[0]
            spec = self._spec
            if nc is None:
                nc = self.__nc
            if cc is None:
                cc = self.__cc
        else:
            num_prompts = self.__parent.prompts.shape[0]
            spec = self.__parent._spec
            if nc is None:
                nc = self.__parent.__nc
            if cc is None:
                cc = self.__parent.__cc
        return ProgressBar(items,total,num_prompts,self._silent,self,keep_line,spec,nc,cc)

    def print_progress(self,index,scale,total,message=None,is_last=True,is_total_last=True,keep_line=True,is_leaf=True,nc=None,cc=None):
        if message is not None:
            if self._message is not None:
                message = '{} {}'.format(self._message,message)
        elif message is None:
            if self._message is not None:
                message = self._message
        if self.__parent is None:
            if nc is None:
                nc = self.__nc
            if cc is None:
                cc = self.__cc
            idx_beg = max(int(scale * index),0)
            idx_pre = max(min(int(scale * (index-1)),idx_beg-1),0)
            idx_end = min(max(int(scale * (index+1)),idx_beg+1),len(self.__prompts))
            self.__prompts[idx_pre:idx_beg] = cc[self.__index_cc]
            if (index+1)==total:
                c = cc[self.__index_cc]
                self.__index_cc = (self.__index_cc + 1) % len(cc)
            else:
                c = nc[self.__index_nc]
                self.__index_nc = (self.__index_nc + 1) % len(nc)
            self.__prompts[idx_beg:idx_end] = c
            percentage = int(float(index+1) * 100 / total)
            message = self._format_message(index,scale,percentage,message)
            is_total_last = is_total_last and (self.__index+1==self.__total)
            end_flag = ''
            if is_last and (is_total_last or not (keep_line and self._keep_line)):
                end_flag = '\n'
            print(message,end=end_flag,flush=True)
        else:
            is_total_last = is_total_last and (self.__index+1==self.__total)
            keep_line = (not is_last) or (keep_line and (is_leaf or self._keep_line or not is_total_last))
            self.__parent.print_progress(index,scale,total,message,is_last,is_total_last,keep_line,False,nc,cc)

    def __next__(self):
        if self.__iterator is None:
            self.__iterator = self.generate_iterator()
        if self._iter_print_message:
            self.print_progress(self.__index,self.__scale,self.__total,None,(self.__index+1==self.__total),nc=self.__nc,cc=self.__cc)
        self.__index += 1

        return next(self.__iterator)

    def __len__(self):
        return self.__total

    def __iter__(self):
        self.__index = -1
        self.__iterator = self.generate_iterator()
        return self

    def generate_iterator(self):
        iterator = None
        if isinstance(self.__items,Iterable):
            iterator = iter(self.__items)
        elif isinstance(self.__items,(Iterator,Generator)):
            iterator = self.__items
        else:
            raise TypeError('`self.__items` must be a instance of Iterable/Iterator/Generator, given {}'.format(type(self.__items)))
        return iterator



def line(message,epochs,iters,epoch=0,iter=0,is_train=True,spec='<{{:0>{}}}{}{{}}>[{{:0>{}}}/{{}},{{:3}}%]',marker='@',newline='auto'):
    assert epochs >= epoch and iters >= iter,'epoch:[{}/{}],iter:[{}/{}]'.format(epochs,epoch,iters,iter)
    head = spec.format(intsize(epochs),marker,intsize(iters)).format(epoch,epochs,iter,iters,iter*100//iters)
    if is_train:
        end = '\n' if (iter == iters and newline.lower() in ['auto','y','yes']) else ''
        print('\r{} {}'.format(head,message),end=end,flush=True)
    else:
        space = len(head)
        size = head.find(marker)
        print('{}|{}> {}'.format(' '*size,'-'*(space-size-2),message),flush=True)


class Line():
    def __init__(self,epochs,iters,spec='<{{:0>{}}}{}{{}}>[{{:0>{}}}/{{}},{{:3}}%]',marker='@'):
        self.epochs = epochs
        self.iters = iters
        self.spec = spec
        self.marker = marker
    def __call__(self,message,epoch=0,iter=0,is_train=True,newline='auto'):
        line(message,self.epochs,self.iters,epoch,iter,is_train,self.spec,self.marker,newline)
