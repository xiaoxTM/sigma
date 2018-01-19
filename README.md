# Software Integration Group Machine Armory
sigma is short for `S`oftware `I`ntegration `G`roup `M`achine `A`rmory. Unlike `Keras` which is class based, `sigma` is a functional fashion framework. That is, all layers are functions rather than classes.

# Support layers
 - actives
   - [x] crelu
   - [x] relu
   - [x] relu6
   - [x] elu
   - [x] selu
   - [x] leaky_relu
   - [x] softmax
   - [x] softplus
   - [x] softsign
   - [x] sigmoid
   - [x] linear
 - base
   - [x] flatten
   - [ ] reshape
 - convolutional
   - [ ] embedding
   - [x] fully_conv [aka. dense]
   - [ ] conv1d
   - [x] conv2d
   - [x] conv3d
   - [x] deconv2d
   - [x] soft_conv2d [aka. deformable convolution]
   - [ ] sepconv2d [aka. separable_conv2d]
 - losses
   - [x] binary_cross_entropy
   - [x] categorical_cross_entropy
   - [x] mean_square_error
   - [ ] winner_takes_all
   - [ ] total_variation_regularize
 - merge
   - [x] concat
   - [x] add
   - [ ] mul
 - normalization
   - [x] instance_norm
   - [x] batch_norm
   - [ ] dropout
 - pools
   - [ ] avg_pool2d
   - [ ] avg_pool2d_global
   - [x] max_pool2d
   - [ ] max_pool2d_global

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
 - graph visualization
   - print no message [status.graph=None]
   - terminal print [status.graph=False]
   - save to file [status.graph=True]
 - load / save and import / export
   - checkpoints
     - [x] load / save
   - weights
     - [x] import_weights / export_weights
   - model
     - [ ] import_model / export_model

# Version explanation
`sigma` version consists of three parts:
- major version indicator
  - increases after great changes
- minor version indicator
  - increases after adding new features
- state version indicator
  - 0 : developing state
  - 1 : testing state
  - 2 : stable state

# Developing progress

```
                                               branch x.x.x.1 / checkout     /-------\
                                             |<---------------------------> | x.x.x.1 | => test
                branch x.x.x.0    /-------\--|        merge x.x.x.1          \-------/
              |----------------> | x.x.x.0 | => devel
              |                   \-------/--|
              |                              | branch x.x.x.2    /-------\
              |                              -----------------> | x.x.x.2 | => stable
              |                                                  \-------/
              |                                                      |
              |                                                      v
   /------\ --|                                                  /-------\
  | master |--------------------------------------------------> |  master |
   \------/                     merge x.x.x.2                    \-------/
```
