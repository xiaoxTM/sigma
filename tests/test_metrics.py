import sys
sys.path.append('/NAS_REMOTE/xiaox/studio/usr/lib')
from sigma.metrics import part_segment_iou,semantic_segment_iou
import os
from sigma import metrics
from Pointnet2_PyTorch.utils import IoU
import numpy as np

bs = 10
ps = 20
cs = 9
preds = np.random.randn(bs,ps,cs)
trues = np.random.randint(0,cs,(bs,ps))

cat2labels = {'cat':[0,8],'dog':[1,6],'rabbit':[2,7],'wolf':[3,5],'cow':[4]}
print(IoU.cal_accuracy_iou(np.argmax(preds,axis=2),trues,cat2labels))
print(metrics.part_segment_iou(cat2labels)(preds,trues))
