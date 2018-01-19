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

data_format='NHWC'
# for channels-last format
#   kernel shape: [row, col, ins, outs] for 2d
# for channel-first (without batch dimension) format
#   kernel shape: [ins, outs, row, col] for 2d
if data_format == 'NHWC':
    axis = -1
else:
    axis = 1
