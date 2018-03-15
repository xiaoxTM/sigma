epsilon = 1e-5
data_format = 'NHWC'
axis = -1

def shape_statistics(shape_list):
    stats = {'nones': [], '-1': []}
    for idx, shape in enumerate(shape_list):
        if shape is None:
            stats['nones'].append(idx)
        elif shape == -1:
            stats['-1'].append(idx)
    return stats
