from .. import ops

def binary_cross_entropy(logits, labels, axis=None, name=None):
    return ops.losses.binary_cross_entropy(axis, name)(logits, labels)

def categorical_cross_entropy(logits, labels, axis=None, name=None):
    return ops.losses.categorical_cross_entropy(axis, name)(logits, labels)

def mean_square_error(logits, labels, axis=None, name=None):
    return ops.losses.mean_square_error(axis, name)(logits, labels)

# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
