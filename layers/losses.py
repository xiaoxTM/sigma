from ..ops import losses

def binary_cross_entropy(logits, labels, axis=None, onehot=True, name=None):
    return losses.binary_cross_entropy(axis, onehot, name)(logits, labels)

def categorical_cross_entropy(logits, labels, axis=None, onehot=True, name=None):
    return losses.categorical_cross_entropy(axis, onehot, name)(logits, labels)

def mean_square_error(logits, labels, axis=None, onehot=True, name=None):
    return losses.mean_square_error(axis, onehot, name)(logits, labels)

def winner_takes_all(logits, labels, axis=None, onehot=True, name=None):
    return losses.winner_takes_all(axis, onehot, name)(logits, labels)

# short alias for each losses
bce = binary_cross_entropy
cce = categorical_cross_entropy
mse = mean_square_error
wta = winner_takes_all
