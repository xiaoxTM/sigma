import numpy as np
import sigma
from sigma.fontstyles import colors
from sigma.utils import intsize

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
            assert total is not None and total > 0, '`items` and `total` cannot be both None'
            self.__total = total
            self.__items = list(range(total))
        else:
            self.__items = list(items)
            self.__total = len(self.__items)
        self.__index = -1
        self.__scale = float(num_prompts) / self.__total
        self._keep_line = keep_line
        if parent is None:
            self.__total_size =sigma.utils.intsize(self.__total)
            self.__nc = [c.encode('utf8') for c in nc]
            self.__cc = [c.encode('utf8') for c in cc]
            self.__index_nc = 0
            self.__index_cc = 0
            self.__prompts = np.asarray([self.__cc[0]] * num_prompts, dtype=np.string_)
            self._spec = spec
            if self._spec is None:
                self._spec = '\r<{{:0>{}}}{}{}> [{{:{}}}, {{:3}}%] {{}}' \
                             .format(self.__total_size,
                                     colors.green('@'),
                                     colors.blue(str(self.__total)),
                                     num_prompts)
            def format_message(index, scale, percentage, message=None):
                idx = max(int(scale * (index+1)),int(scale * index)+1)
                if message is None:
                    message = ''
                return self._spec.format(self.__index+1, self.__prompts[:idx].tostring().decode('utf-8'), percentage, message)
            self._format_message = format_message
        self._iter_print_message = not silent
        self._silent = silent
        self._message = None

    def set_message(self, message):
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

    def sub(self, items, total:int=None, keep_line:bool=True):
        self._iter_print_message = False
        if self.__parent is None: # itself is parent
            num_prompts = self.__prompts.shape[0]
            spec = self._spec
            nc = self.__nc
            cc = self.__cc
        else:
            num_prompts = self.__parent.prompts.shape[0]
            spec = self.__parent._spec
            nc = self.__parent.__nc
            cc = self.__parent.__cc
        return ProgressBar(items,total,num_prompts,self._silent,self,keep_line,spec,nc,cc)

    def print_progress(self,index,scale,total,message=None,is_last=True,is_total_last=True,keep_line=True,is_leaf=True):
        if message is not None:
            if self._message is not None:
                message = '{} {}'.format(self._message, message)
        elif message is None:
            if self._message is not None:
                message = self._message
        if self.__parent is None:
            idx_beg = max(int(scale * index), 0)
            idx_pre = max(min(int(scale * (index-1)), idx_beg-1), 0)
            idx_end = min(max(int(scale * (index+1)), idx_beg+1), len(self.__prompts))
            self.__prompts[idx_pre:idx_beg] = self.__cc[self.__index_cc]
            if (index+1)==total:
                c = self.__cc[self.__index_cc]
                self.__index_cc = (self.__index_cc + 1) % len(self.__cc)
            else:
                c = self.__nc[self.__index_nc]
                self.__index_nc = (self.__index_nc + 1) % len(self.__nc)
            self.__prompts[idx_beg:idx_end] = c
            percentage = int(float(index+1) * 100 / total)
            message = self._format_message(index, scale, percentage, message)
            is_total_last = is_total_last and (self.__index+1==self.__total)
            end_flag = ''
            if is_last and (is_total_last or not (keep_line and self._keep_line)):
                end_flag = '\n'
            print(message, end=end_flag, flush=True)
        else:
            is_total_last = is_total_last and (self.__index+1==self.__total)
            keep_line = (not is_last) or (keep_line and (is_leaf or self._keep_line or not is_total_last))
            self.__parent.print_progress(index, scale, total, message, is_last, is_total_last, keep_line,False)

    def __next__(self):
        if self._iter_print_message:
            self.print_progress(self.__index, self.__scale, self.__total, None, (self.__index+1==self.__total))
        self.__index +=1

        if self.__index < self.__total:
            return self.__items[self.__index]
        else:
            raise StopIteration

    def __len__(self):
        return self.__total

    def __iter__(self):
        self.__index = -1
        return self



def line(message,epochs,iters,epoch=0,iter=0,is_train=True,spec='<{{:0>{}}}{}{{}}>[{{:0>{}}}/{{}},{{:3}}%]',marker='@',newline='auto'):
    assert epochs >= epoch and iters >= iter
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
        
# if __name__ == '__main__':
#     from time import sleep
#     ## one hierarchy
#     print('one hierarchy')
#     p = ProgressBar(total=400, nc='x+*+', cc='><')
#     p._keep_line=False
#     for n in p:
#         sleep(0.01)
#         p.set_message('n:{}'.format(n))
#     print('two hierarchies keep line')
#     ## two hierarchies
#     pb = ProgressBar(range(10), num_prompts=10, cc='><', keep_line=True)
#     spb = pb.sub(range(10))
#     spb2 = pb.sub(range(20))
#     for i in pb:
#         pb.set_message('i:{}'.format(i))
#         for j in spb:
#             spb.set_message('j:{}'.format(j))
#         for k in spb2:
#             spb2.set_message('k:{}'.format(k))
#             sleep(0.1)
#     print('two hierarchies not keep line')
#     ## two hierarchies
#     pb = ProgressBar(range(10), num_prompts=10, cc='><', keep_line=False)
#     spb = pb.sub(range(10))
#     spb2 = pb.sub(range(20))
#     for i in pb:
#         pb.set_message('i:{}'.format(i))
#         for j in spb:
#             spb.set_message('j:{}'.format(j))
#         for k in spb2:
#             spb2.set_message('k:{}'.format(k))
#             sleep(0.1)
#     print('three hierarchies')
#     ## three hierarchies
#     pb = ProgressBar(enumerate([2,3,4,5,6]), num_prompts=20, cc='>^<')
#     pb._keep_line = False
#     spb = pb.sub(range(3))
#     spb._keep_line = False
#     sspb = spb.sub(range(10))
#     #sspb._keep_line = False
#     # keep_line=False from the parent progressbar covers the value of keep_line from sub progressbar
#     for i, v in pb:
#         for j in spb:
#             for k in sspb:
#                 sleep(0.01)
