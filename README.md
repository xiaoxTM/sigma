# Software Integration Group Machine Armory
sigma is short for `S`oftware `I`ntegration `G`roup `M`achine `A`rmory. Unlike `Keras` which is class based, `sigma` is a functional fashion framework. That is, all layers are function rather than class.

# Support layers
 - actives
   - crelu
   - relu
   - relu6
   - elu
   - selu
   - leaky_relu
   - softmax
   - softplus
   - softsign
   - sigmoid
   - linear
 - base
   - flatten
   - reshape (not tested)
 - convolutional
   - embedding (not tested)
   - fully_conv [aka. dense]
   - conv1d (not tested)
   - conv2d
   - conv3d (not tested)
   - deconv2d
   - soft_conv2d [aka. deformable convolution]
 - losses
   - binary_cross_entropy
   - categorical_cross_entropy
   - mean_square_error
   - winner_takes_all (not tested)
 - merge
   - concat
   - add (not tested)
   - mul (not tested)
 - normalization
   - batch_norm
   - dropout (not tested)
 - pools
   - avg_pool2d
   - avg_pool2d_global
   - max_pool2d
   - max_pool2d_global

# Advantages
 - seamless with original tensorflow library
 - with statement to set default value
   - e.g., with sigma.defaults(parameters=values)
 - deformable convolutional layers. with gather element mode.
   see [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) for details.
   - naive : just cast float location to int
   - nearest : get the nearest location
   - floor : get the floor location
   - ceil : get the ceil location
   - bilinear : bilinear interpolation
