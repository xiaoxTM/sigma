# record status of sigma
# e.g.,:
#    is_training
# this variable is useful for layers like:
#    - batch-norm
is_training = True
data_format='NHWC'
if data_format == 'NHWC':
    axis = -1
else:
    axis = 1
