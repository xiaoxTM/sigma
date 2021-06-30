import numpy as np
from sigma import parse_params
from sigma.fontstyles import colors
from sklearn import metrics
import functools

def accuracy_score(*args, **kwargs):
    score = functools.partial(metrics.accuracy_score, *args, **kwargs)
    def _score(preds, trues):
        return {'acc':score(trues, preds)}
    return _score

def balanced_accuracy_score(*args, **kwargs):
    score = functools.partial(metrics.balanced_accuracy_score, *args, **kwargs)
    def _score(preds, trues):
        return {'bac':score(trues, preds)}
    return _score

def roc_auc_score(*args, **kwargs):
    score = functools.partial(metrics.roc_auc_score, *args, **kwargs)
    def _score(preds, trues):
        return {'roc':score(trues, preds)}
    return _score

def average_precision_score(*args, **kwargs):
    score = functools.partial(metrics.average_precision_score, *args, **kwargs)
    def _score(preds, trues):
        return {'ap':score(true, preds)}
    return _score

def f1_score(*args, **kwargs):
    score = functools.partial(metrics.f1_score, *args, **kwargs)
    def _score(preds, trues):
        return {'f1':score(trues, preds)}
    return _score

def precision_score(*args, **kwargs):
    score = functools.partial(metrics.precision_score, *args, **kwargs)
    def _score(preds, trues):
        return {'pre':score(trues, preds)}
    return _score

def recall_score(*args, **kwargs):
    score = functools.partial(metrics.recall_score, *args, **kwargs)
    def _score(preds, trues):
        return {'rec':score(trues, preds)}
    return _score


__metrics__ = {'accuracy': accuracy_score,
               'acc': accuracy_score,
               'balanced_accuracy': balanced_accuracy_score,
               'bac': balanced_accuracy_score,
               'roc': roc_auc_score,
               'ap': average_precision_score,
               'f1': f1_score,
               'precision': precision_score,
               'pre': precision_score,
               'recall': recall_score,
               'rec': recall_score}

def part_segment_iou(logits, trues, labels2name, name2labels, ignore_not_exist_labels=True):
    ''' iou for part segmentation
        instance IoU:
             correct-segmented-points / (num-points * 2 - correct-segmented-points)
        e.g., ground truth:[5,4] prediction: [5,6]
             intersection = corrected-segmented-points = (5 at [0]) = 1
             unions = [5,4,5,6] - [5] = 4 - 1 = 3
             IoU = 1 / 3
        class IoU:
             for class c:
                 intersection = (logits==c && trues==c)
                 unions       = (logits==1 || trues==c)
                 IoU = intersection / unions
        balanced accuracy score (bac):
             for class c:
                 c_trues = trues == c
                 c_preds   =   preds[c_trues]
        e.g., [[4,3],[4,6]] -> pred, [[4,5],[3,4]] -> true
             for class 4: true positive = (4,3)[0]
                          false negative =(4,6)[1]
                          false positive =(4,6)[0]
              acc = true-positive / (true-positive + false-negative) = 1 / 2
              NOTE that, the false positive (4,6)[0] is ignored

    '''
    # logits: list of [batch-size, num-points, num-part-classes]
    # true: list of [batch-size, num-points]
    # labels2name: dict that mapping label to object name
    # name2labels: dict that mapping object name to label
    # ignore_not_exist_labels: bool, ignore labels that not exists in true if `True`, other wise set to `0`
    # NOTE: labels inside the part of an object must be continuous
    #       e.g., {'Airplane': [34,35,36]}
    #       labels that not appear neither in logits nor in true will also ignored
    num_samples, num_points, num_part_classes = logits.shape
    # class_iou: {'Airplane': np.array([iou_0, iou_1, iou_2, ...]),
    #             'Motorbike': np.array([iou_0, iou_1, ...]),
    #              ...
    #            }
    part_class_ious = {name:[] for name in name2labels.keys()}
    total_instance_ious = []
    total_correct = 0
    total_points = num_samples * num_points
    total_seen_objects = np.zeros(num_part_classes)
    total_correct_object = np.zeros(num_part_classes)


    for object_name, part_labels in name2labels.items():
        # for each part, segment them
        # get the part columns from logits
        #print('object name:', object_name)
        indices = []
        for part_label in part_labels:
            hit = np.where(trues==part_label)[0] # only need rows
            if len(hit) > 0:
                indices.extend(hit)
        if len(indices) == 0:
            part_class_ious.pop(object_name)
        else:
            indices = np.unique(indices)
            # logit: [num-samples, num-points, num-parts]
            logit = np.stack([logits[indices,:,part_label] for part_label in part_labels],axis=2)
            # logit: [num-samples, num-points]
            pred = np.argmax(logit,axis=2)+part_labels[0]
            #print('pred:', pred)
            true = trues[indices,:]
            #print('true:', true)
            part_iou = []
            for part_label in part_labels:
                # ground truth: [num-instances, num-points]
                ground_truth = np.where(true==part_label,1,0)
                # positive = [num-instances]
                seen_classes = ground_truth.sum(axis=1)
                # part-inters: [num-instances, num-points]
                part_inters = np.where((true==part_label)&(pred==part_label),1,0)
                # part-unions: [num-instances, num-points]
                part_unions = np.where((true==part_label)|(pred==part_label),1,0)
                if ignore_not_exist_labels:
                    available_sample_indices = np.where(seen_classes!=0)[0]
                    part_inters = part_inters[available_sample_indices,:]
                    part_unions = part_unions[available_sample_indices,:]
                if part_unions.shape[0] > 0:
                    #                                     [num-instances]
                    part_iou.extend(part_inters.sum(axis=1) / part_unions.sum(axis=1))
                total_seen_objects[part_label] += seen_classes.sum()
                total_correct_object[part_label] += part_inters.sum()
            if len(part_iou) > 0:
                part_class_ious[object_name] = part_iou
            else:
                part_class_ious.pop(object_name)
            #true_positive = np.where(pred==true,1,0)
            # instance IoU
            #instance_inters = true_positive.sum(axis=1)
            #instance_unions = num_points*2 - instance_inters
            #total_instance_ious.extend(instance_inters / instance_unions)
            total_instance_iou.extend(part_iou)
            total_correct += np.where(pred==true,1,0).sum()

    total_class_ious = {k:0 for k in part_class_ious.keys()}
    for object_name, values in part_class_ious.items():
        total_class_ious[object_name] = np.mean(values)
    class_miou = np.mean(list(total_class_ious.values()))
    instance_miou = np.mean(total_instance_ious)

    acc = total_correct / float(total_points)
    acs = []
    for correct_object, seen_object in zip(total_correct_object, total_seen_objects):
        if seen_object > 0:
            acs.append(correct_object / float(seen_object))
    bac = np.mean(acs)

    return acc, bac, class_miou, instance_miou, total_class_ious


def shapenet_part_iou(logits, trues, ignore_not_exist_labels=True):
    # logits: list of [batch-size, num-points, num-part-classes]
    # true: list of [batch-size, num-points]
    # labels2name: dict that mapping label to object name
    # name2labels: dict that mapping object name to label
    # NOTE: labels inside the part of an object must be continuous
    #       e.g., {'Airplane': [34,35,36]}
    name2labels = {'Earphone': [16, 17, 18],
                   'Motorbike': [30, 31, 32, 33, 34, 35],
                   'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11],
                   'Laptop': [28, 29],
                   'Cap': [6, 7],
                   'Skateboard': [44, 45, 46],
                   'Mug': [36, 37],
                   'Guitar': [19, 20, 21],
                   'Bag': [4, 5],
                   'Lamp': [24, 25, 26, 27],
                   'Table': [47, 48, 49],
                   'Airplane': [0, 1, 2, 3],
                   'Pistol': [38, 39, 40],
                   'Chair': [12, 13, 14, 15],
                   'Knife': [22, 23]}
    labels2name = {}
    for name, labels in name2labels.items():
        for label in labels:
            labels2name[label] = name
    return part_segment_iou(logits, trues, labels2name, nametlabels, ignore_not_exist_labels)


def semantic_segment_iou(logits, trues):
    # logits: batch-size, num-points, num-classes
    # trues: batch-size, num-points

    num_samples, num_points, num_classes = logits.shape
    logits = np.argmax(logits, 2) # batch-size, num-points

    total_seen = np.zeros(num_classes)
    total_inter = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    for c in range(num_classes):
        total_seen[c] += np.sum(trues==c)
        total_inter[c] += np.sum((logits==c)&(trues==c))
        total_union[c] += np.sum((logits==c)|(trues==c))

    acc = np.sum(logits == trues) / float(num_samples * num_points)
    iou = []
    acs = []
    # get rid of classes that does not appear in ground truth
    for inter, union, seen in zip(total_inter, total_union, total_seen):
        if seen > 0:
            iou.append(inter / float(union))
            acs.append(inter / float(seen))
    miou = np.mean(iou)
    bac = np.mean(acs)

    return acc, bac, miou


def metric_class_iou(preds, trues, num_classes):
    total_seen = np.zeros(num_classes)
    total_inter = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    for c in range(num_classes):
        total_seen[c] += np.sum(trues==c)
        total_inter[c] += np.sum((preds==c)&(trues==c))
        total_union[c] += np.sum((preds==c)|(trues==c))

    iou = []
    # get rid of classes that does not appear in ground truth
    for inter, union, seen in zip(total_inter, total_union, total_seen):
        if seen > 0:
            iou.append(inter / float(union))
        else:
            iou.append(-1)
    # IoU for each class
    return iou


def metric_class_miou(preds, trues, num_classes):
    iou = metric_iou(preds, trues, num_classes)
    iou = [i for i in iou  if i >= 0]
    # class mIoU
    return np.mean(iou)


def benchmark_miou(preds, trues, num_classes):
    # mIoU for a batch
    batch_size, num_points = preds.shape
    shape_iou = []
    for batch in range(batch_size):
        parts = np.arange(num_classes)
        part_iou = []
        for part in parts:
            inter = np.sum(np.logical_and(preds[batch]==part, trues[batch]==part))
            union = np.sum(np.logical_or(preds[batch]==part, trues[batch]==part))
            part_iou.append(1 if union == 0 else float(inter) / union)
        shape_iou.append(np.mean(part_iou))
    return shape_iou


def get(metric):
    if metric is None or callable(metric):
        return metric
    elif isinstance(metric, str):
        metric = metric.strip()
        if metric in ['','null','none']:
            return None
        metric_type, params = parse_params(metric)
        metric_type = metric_type.lower()
        assert metric_type in __metrics__.keys(), 'metric type {} not support'.format(metric_type)
        return __metrics__[metric_type](**params)
    else:
        raise TypeError('cannot convert type {} into metric function'.format(colors.red(type(metric))))

def register(key, metric):
    assert key is not None and metric is not None, 'both key and metric can not be none'
    global __metrics__
    assert key not in __metrics__.keys(), 'key {} already registered'.format(key)
    assert callable(metric), 'metric must be function, given {}'.format(type(metric))
    __metrics__.update({key:metric})


if __name__ == '__main__':
    batch_size = 3
    num_points = 2
    num_classes = 6 #(0,1,2), (3,4,5)
    for i in range(1000):
        logits = np.random.rand(batch_size, num_points, num_classes)
        trues = np.random.randint(0,3, (batch_size,num_points)) + np.random.randint(0,2, (batch_size,1)) * 3
        #print('logits:\n',logits)
        #print('trues:\n',trues)
        #acc, bac, miou = semantic_segment_iou(logits, trues)
        #print('acc:',acc,'/ bac:',bac,'/ mIoU:',miou)
        print('>>>>>>>>>>><<<<<<<<<<<')
        label2name = {0:'a', 1:'a', 2:'a', 3:'b', 4:'b', 5:'b'}
        name2label = {'a':[0,1,2], 'b':[3,4,5]}
        print("'a':[0,1,2], 'b':[3,4,5]}")
        acc, bac, class_miou, instance_miou, total_class_ious = part_segment_iou(logits, trues, label2name, name2label)
        print('acc:',acc,'/ bac:',bac,'/ class mean iou:',class_miou, '/ instance miou:',instance_miou)

        # code borrowed from pointnet.pytorch
        num_part = num_classes
        seg_classes = name2label
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        # num part: total number of labels for shapenet parts
        total_seen_class = [0] * num_part #[0 for _ in range(num_part)]
        total_correct_class = [0] * num_part #[0 for _ in range(num_part)]
        # shape_ious: [Airplane:[], Rocket:[], ...]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        # global `seg_label_to_cat` to local `seg_label_to_cat`
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for key, labels in seg_classes.items():
            for label in labels:
                seg_label_to_cat[label] = key

        cur_pred_val_logit = logits
        cur_pred_val = np.zeros((batch_size, num_points)).astype(np.int32)
        # true: [batch-size, num-part]
        # i.e., [batch-size, 50]
        true = trues
        for i in range(batch_size):
            #cat: name of the category, e.g., `Airplane` or `Motorbike`
            # true[i,j] = true[i,k]
            cat = seg_label_to_cat[true[i, 0]]
            # the `i-th` logits in current batch
            # logits: num-points, num-parts
            logit = cur_pred_val_logit[i, :, :]
            # logits[:, seg_classes[cat]] indicates all [part-]classes in category `cat`
            # note that, classes in `cat` is a subset of total classes(50)
            # i.e., object classes (parts-label) <= total classes (object-label) (50)
            #                                                             `+` convert parts-label to object label
            # NOTE that, part-labels belong to the same object are continuous, e.g., 12, 13,14
            cur_pred_val[i, :] = np.argmax(logit[:, seg_classes[cat]], 1) + seg_classes[cat][0]
        correct = np.sum(cur_pred_val == true) # AND
        total_correct += correct
        total_seen += (batch_size * num_points)

        for l in range(num_part):
            # all classes that exist in ground truth, i.e., total positive points
            total_seen_class[l] += np.sum(true == l)
            # total true positive points for each parts
            total_correct_class[l] += (np.sum((cur_pred_val == l) & (true == l)))

        for i in range(batch_size):
            segp = cur_pred_val[i, :]
            segl = true[i, :]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0] * (len(seg_classes[cat]))
            for l in seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (
                        np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        print('acc:',total_correct / float(total_seen))
        print('bac:',np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)))
        print('class mean iou:',mean_shape_ious)
        print('instance mean iou:',np.mean(all_shape_ious))
