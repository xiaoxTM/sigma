import visdom
from sigma.utils import merge_args

class Webviz():
    def __init__(self,*args,**kwargs):
        super(Webviz,self).__init__()
        self._wins = {}
        self._visdom = visdom.Visdom(*args, **kwargs)

    def autoplot(self, fun, key, updatable, *args,**kwargs):
        wargs = merge_args(args, kwargs, fun)
        if not updatable:
            win = wargs.get('win',None)
            opts = wargs.get('opts',{})
            wargs.pop('name', None)
            wargs.pop('update',None)
            if opts is None:
                opts = {}
            if opts.get('title', None) is None:
                opts.update({'title':key})
            if key not in self._wins.keys():
                wargs.update({'win':None})
                wargs.update({'opts':opts})
                win = fun(**wargs)
                self._wins.update({key:[win]})
            else:
                win = self._wins[key][0]
                wargs.update({'win':win})
                fun(**wargs)
        else:
            update = wargs.get('update',None)
            assert update in ['append','remve','replace'], f'update:{update} value not support'
            win = wargs.get('win',None)
            name = wargs.get('name',key)
            assert name is not None, 'name can not be None'
            opts = wargs.get('opts',{})
            if opts is None:
                opts = {}
            if opts.get('title', None) is None:
                opts.update({'title':key})
            if key not in self._wins.keys():
                wargs.update({'update':None,'win':None})
                wargs.update({'opts':opts})
                win = fun(**wargs)
                self._wins.update({key:[win,name]})
            else:
                win = self._wins[key][0]
                if name not in self._wins[key][1:]:
                    update = 'new'
                    self._wins[key].append(name)
                wargs.update({'win':win})
                fun(**wargs)

    def line(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.line, key, True, *args, **kwargs)
    # markersymbol: https://plotly.com/python/marker-style/
    # markersize
    def scatter(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.scatter, key, True, *args, **kwargs)

    def image(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.image, key, False, *args, **kwargs)

    def images(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.images, key, False, *args, **kwargs)

    def text(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.text, key, False, *args, **kwargs)

    def audio(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.audio, key, False, *args, **kwargs)

    def video(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.video, key, False, *args, **kwargs)

    def stem(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.stem, key, False, *args, **kwargs)

    def bar(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.bar, key, False, *args, **kwargs)

    def histogram(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.histogram, key, False, *args, **kwargs)

    def boxplot(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.boxplot, key, False, *args, **kwargs)

    def surf(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.surf, key, False, *args, **kwargs)

    def contour(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.contour, key, False, *args, **kwargs)

    def quiver(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.quiver, key, False, *args, **kwargs)

    def mesh(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.mesh, key, False, *args, **kwargs)

    def dual_axis_lines(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.dual_axis_lines, key, False, *args, **kwargs)

    def heatmap(self, key, *args, **kwargs):
        return self.autoplot(self._visdom.heatmap, key, False, *args, **kwargs)

    #   matplot(self,plot,opts=None,env=None,win=None)
    def matplot(self, key, plt_or_figure, env=None, opts=None):
        win=self._visdom.matplot(plt_or_figure, opts=opts, env=env, win=self._wins.get(key,None))
        self._wins.update({key:win})

    def close(self):
        self._visdom.close()
