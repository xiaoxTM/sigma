from . import ngan

# def build_gan(network, ninput, rdata, **kwargs):
#     """ network:
#         Attributes
#         ==========
#         network : string
#                   network name, currently supported: `toy`, `mnist`
#         ninput : int
#                  number of inputs for random noise
#         rdata : tf.tensor
#                 input tensor from `build`
#     """
#     module = import_module('.{}'.format(network), __name__)
#     outs = rdata.get_shape().as_list()[1:]
#
#     generate, discriminate, options = module.get(outs=outs, **kwargs)
#
#     # x  or
#     # x, label
#     fdata = generate([options.get('batch-size', 64), ninput])
#
#     if not ops.helper.is_tensor(fdata):
#         # not tensor, should be list /tuple
#         # in case sample labels are provided
#         # as extra information
#         # regression loss
#
#         nclass = kwargs.get('nclass', None)
#         if nclass is None:
#             raise ValueError('module requires `nclass` which is not provided to `build_gan`')
#         elif not isinstance(nclass, int):
#             raise ValueError('module requires `nclass` as `int`, given {}'.format(type(x)))
#         labels = tf.placeholder(tf.int32, [options.get('batch-size', 64), 1])
#
#         dreal = discriminate(rdata)
#         dfake = discriminate(fdata[0], reuse=True)
#
#         dloss =  tf.reduce_mean(dfake[0]) - tf.reduce_mean(dreal[0])
#         gloss = -tf.reduce_mean(dfake[0])
#         dloss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, depth=nclass),
#                                                                         logits=dreal[1]))
#         gloss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(fdata[1], depth=nclass),
#                                                                         logits=dfake[1]))
#         options['labels'] = labels
#
#     else:
#         dreal = discriminate(rdata)
#         dfake = discriminate(fdata, reuse=True)
#         dloss =  tf.reduce_mean(dfake) - tf.reduce_mean(dreal)
#         gloss = -tf.reduce_mean(dfake)
#
#     options['generate-data'] = fdata
#     options['generate-data-loss'] = gloss
#     options['generate-data-output'] = dfake
#     options['real-data-output'] = dreal
#     options['loss'] = dloss
#
#     return generate, discriminate, options
#
#
# def build_wgan_wc(network, ninput, rdata, **kwargs):
#
#     _, _, options = build_gan(network, ninput, rdata, **kwargs)
#
#     gparams = tf.get_collection('generator')
#     dparams = tf.get_collection('discriminator')
#
#     discriminator = tf.train.RMSPropOptimizer(learning_rate=options.get('discriminator-lr', 0.0001))\
#                                               .minimize(options['loss'], var_list=dparams)
#     generator = tf.no_op()
#     if data_bland is not None:
#         generator = tf.train.RMSPropOptimizer(learning_rate=options.get('generator-lr', 0.0001))\
#                                               .minimize(options['generate-data-loss'], var_list=gparams)
#     clipper = tf.assign(dparams, tf.clip_by_value(dparams, options.get('clip-min', 0), options.get('clip-max', 1)))
#
#     options['clipper'] = clipper
#
#     return goptimizer, doptimizer, options
#
# def build_wgan_gp(network, ninput, rdata, **kwargs):
#
#     generate, discriminate, options = build_gan(network, ninput, rdata, **kwargs)
#     fdata = options['generate-data']
#     dloss = options['loss']
#
#     # if fdata is not tf.Tensor
#     # since it is difficult to assign labels
#     # to interpolates, we ignore gradient penalty
#     if ops.helper.is_tensor(fdata):
#         gshape = fdata.get_shape().as_list()
#         alpha = tf.random_uniform(shape=(gshape[0],), minval=0.0, maxval=1.0)
#         alpha = tf.reshape(alpha, [-1] + [1] * (len(gshape)-1))
#         interpolates = alpha * rdata + (1-alpha) * fdata
#         dinterpolation = discriminate(interpolates, reuse=True)
#         gradients = tf.gradients(dinterpolation, [interpolates])[0]
#         slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
#         penalty = tf.reduce_mean((slopes - 1) ** 2)
#         dloss += (options.get('lambda', 0.2) * penalty)
#
#     gparams = tf.get_collection('generator')
#     # for gp in gparams:
#     #     print(gp.name, gp.get_shape())
#     dparams = tf.get_collection('discriminator')
#     doptimizer = tf.train.AdamOptimizer(learning_rate=options.get('discriminator-lr', 0.0001),
#                                         beta1=options.get('discriminator-beta1', 0.5),
#                                         beta2=options.get('discriminator-beta2', 0.9)
#                                         ).minimize(dloss, var_list=dparams)
#     goptimizer = tf.no_op()
#     if len(gparams) > 0:
#         goptimizer = tf.train.AdamOptimizer(learning_rate=options.get('generator-lr', 0.0001),
#                                             beta1=options.get('generator-beta1', 0.5),
#                                             beta2=options.get('generator-beta2', 0.9)
#                                             ).minimize(options['generate-data-loss'], var_list=gparams)
#     options['loss'] = dloss
#     return goptimizer, doptimizer, options
#
# def build_dcgan(network, ninput, rdata, **kwargs):
#     pass
#
# def build(network, mode, ninput, shape, **kwargs):
#     rdata = tf.placeholder(tf.float32, shape=[None, *shape])
#     return rdata, eval('build_{}(network, ninput, rdata, **kwargs)'.format(mode.replace('-', '_'))) # wgan-xx -> wgan_xx
