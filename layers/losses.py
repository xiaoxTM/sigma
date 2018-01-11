from ..ops import losses
from ..ops import regularizers

def binary_cross_entropy(logits, labels, axis=None, onehot=True, name=None):
    return losses.binary_cross_entropy(axis, onehot, name)(logits, labels)

def categorical_cross_entropy(logits, labels, axis=None, onehot=True, name=None):
    return losses.categorical_cross_entropy(axis, onehot, name)(logits, labels)

def mean_square_error(logits, labels, axis=None, onehot=True, name=None):
    return losses.mean_square_error(axis, onehot, name)(logits, labels)

def winner_takes_all(logits, labels, axis=None, onehot=True, name=None):
    return losses.winner_takes_all(axis, onehot, name)(logits, labels)

def total_variation_regularize(x):
    return regularizers.total_variation_regularizer(x.get_shape().as_list())(x)

# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
wta = winner_takes_all
tvr = total_variation_regularize
