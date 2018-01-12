from .layers import defaults

__version__ = '0.1'

data_format='NHWC'
# for channels-last format
#   kernel shape: [row, col, ins, outs] for 2d
# for channel-first (without batch dimension) format
#   kernel shape: [ins, outs, row, col] for 2d
if data_format == 'NHWC':
    axis = -1
else:
    axis = 1
