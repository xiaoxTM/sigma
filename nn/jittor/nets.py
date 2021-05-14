import nn

class LocalResponseNorm(nn.Module):
    def __init__(self,
                 local_size=1,
                 alpha=1e-4,
                 beta=0.75,
                 cross_channels=False):
        super(LocalResponseNorm,self).__init__()
        self.cross_channels = cross_channels
        self.alpha = alpha
        self.beta = beta
        if self.cross_channels:
            self.average = nn.AvgPool3d(kernel_size=(local_size,1,1),
                                        stride=1,
                                        padding=int((local_size-1.0)//2))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size-1.0)//2))

    def forward(self, x):
        if self.cross_channels:
            div = self.average(x.pow(2).unsqueeze(1)).squeeze(1)
        else:
            div = self.average(x.pow(2))

        div.mul_(self.alpha)
        div.add_(2.0)
        div.pow_(self.beta)
        x.div_(div)
        return x


def vgg_block(cin,cout,
              ksize=3,
              stride=1,
              padding=0,
              use_lrn=True,
              psize=2):
    layers = [nn.Conv2d(cin,cout,ksize,
                        stride=stride,
                        padding=padding),
              nn.ReLU(inplace=True)]
    if use_lrn:
        layers.append(LocalResponseNorm())
    if psize > 0:
        layers.append(nn.MaxPool2d(psize))
    return layers

class VGGBase(nn.Module):
    def __init__(self, with_classifier=True, cfg):
        super(VGGBase,self).__init__()
        s1,s2,s3,s4 = cfg.get('filters',[96,256,512,512*6*6])
        self.features = nn.Sequential(
                *vgg_block(3,s1,**cfg['conv1']),
                *vgg_block(s1,s2,**cfg['conv2']),
                *vgg_block(s2,s3,padding=1,use_lrn=False,psize=0),
                *vgg_block(s3,s3,padding=1,use_lrn=False,psize=0),
                *vgg_block(s3,s3,padding=1,use_lrn=False,**cfg.get('conv5',{}))
                )
        if with_classifier:
            self.classifier = nn.Sequential(
                    nn.Linear(s4,4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096,4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(4096,cfg.get('num_classes')))
        self.with_classifier = with_classifier

    def forward(self,x):
        x = self.features(x)
        if self.with_classifier:
            x = self.classifier(x)
        return x

def vgg_f(num_classes, with_classifier=True):
    return VGGBase(with_classifier,
                   {'filters':[64,256,256,256*6*6],
                    'conv1':{
                        'ksize':11,
                        'stride':4
                        },
                    'conv2':{
                        'ksize':5,
                        'stride':1,
                        'padding':2
                        },
                    'num_classes':num_classes})

def vgg_m(num_classes):
    return VGGBase(with_classifier,
                   {'filters':[96,256,512,512*6*6],
                    'conv1':{
                        'ksize':7,
                        'stride':2
                        },
                    'conv2':{
                        'ksize':5,
                        'stride':2,
                        'padding':1
                        },
                    'num_classes':num_classes})

def vgg_s(num_classes):
    return VGGBase(with_classifier,
                   {'filters':[96,256,512,512*5*5],
                    'conv1':{
                        'ksize':7,
                        'stride':2,
                        'psize':3
                        },
                    'conv2':{
                        'ksize':5,
                        'stride':1,
                        'padding':1,
                        'use_lrn':False
                        },
                    'conv5':{
                        'psize':3
                        },
                    'num_classes':num_classes})

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, with_classifier):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        if with_classifier:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        self.with_classifier = with_classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.with_classifier:
            x = self.classifier(x)
        return x


def alexnet(num_classes, with_classifier=True):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(num_classes, with_classifier)
    return model
