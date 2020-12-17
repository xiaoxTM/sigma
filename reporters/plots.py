from sigma.metrics.evaluation import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def heatmap(preds,trues,labels=None,normalize=None,rotation=45,annotate=False,cbargs=None,sargs={},**kwargs):
    hits = confusion_matrix(preds,trues,normalize=normalize)
    fig,ax = plt.subplots(**kwargs)
    ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    im = ax.imshow(hits,**sargs)
    if cbargs is not None:
        fig.colorbar(im,**cbargs)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(),rotation=rotation,ha='center',rotation_mode='anchor')
    if annotate:
        for i in range(hits.shape[0]):
            for j in range(hits.shape[1]):
                text = ax.text(j,i,'{:0.2F}'.format(hits[i,j]),
                           ha='center',va='center',color='w')
    return fig,ax

def scatter(preds,trues,labels=None,size=10,xticks=None,grid=True,pm='+',tm='o',pargs={},targs={},**kwargs):
    fig,ax = plt.subplots(**kwargs)
    x = np.arange(len(preds))
    ax.scatter(x,preds,marker=pm,label='pred',**pargs)
    ax.scatter(x,trues,marker=tm,label='true',**targs)
    if labels is not None:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(grid)
    ax.legend()
    return fig,ax

def walk(x,y,endpoint_marker=None,labels=None,xlabel=None,ylabel=None,**kwargs):
    fig = plt.figure(**kwargs)
    ax = fig.gca()
    ax.plot(x,y)
    if endpoint_marker:
        for ex,ey in zip(x[-1],y[-1]):
            ax.scatter(ex,ey,marker=endpoint_marker)
    if labels:
        ax.legend(labels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return fig,ax
