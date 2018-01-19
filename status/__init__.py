# record status of sigma
# e.g.,:
#    is_training
# this variable is useful for layers like:
#    - batch-norm
is_training = True

# interstatus for visualization
# must be one of [None, True, False]
#    None: no output
#    True: print layers to graph
#          will change to instance of pydot.Dot
#    False:print layers to terminal
graph = None

data_format = 'NHWC'
# for channels-last format
#   kernel shape: [row, col, ins, outs] for 2d
# for channel-first (without batch dimension) format
#   kernel shape: [ins, outs, row, col] for 2d

axis = -1

def set_data_format(dformat):
    if dformat not in ['NHWC', 'NCWH']:
        raise ValueError('`data_format` must be `NHWC` or `NCWH`. given {}'
                         .format(dformat))
    global axis
    global data_format
    data_format = dformat
    if data_format == 'NHWC':
        axis = -1
    else:
        axis = 1
