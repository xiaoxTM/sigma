import sigma
from .. import layers


def _encode_layer(x, nout,
                  reuse=False,
                  scope=None):
    x = layers.actives.leaky_relu(x)
    x = layers.convs.conv2d(x, nout, kshape=5, stride=2)
    x = layers.norms.batch_norm(x)
    return x


def _decode_layer(x, nout, outshape, enc, ids, bank_size, 
                  dropout=False,
                  last_layer=False,
                  norm_type='instance',
                  reuse=False,
                  scope=None):
    x = layers.actives.relu(x, reuse=reuse, scope=scope)
    x = layers.convs.deconv2d(x, nout, outshape, reuse=reuse, scope=scope)
    if not last_layer:
        if norm_type == 'instance':
            x = layers.norms.conditional_instance_norm(x, ids, bank_size, reuse=reuse, scope=scope)
        elif norm_type == 'batch':
            x = layers.norms.batch_norm(x, reuse=reuse, scope=scope)
    if dropout:
        x = layers.norms.dropout(x, 0.5, reuse=reuse, scope=scope)
    if enc is not None:
        x = layers.merge.concat([x, enc], reuse=reuse, scope=scope)
    return x


""" encoder of calligraphy networks
"""
def _encoder(inputs, base, reuse, scope):
    enc_layers = {}
    enc_layers['e0'] = inputs
    x = layers.convs.conv2d(inputs, base, reuse=reuse, scope=scope)
    enc_layers['e1'] = x
    x = _encode_layer(x, base*2, reuse=reuse, scope=scope)
    enc_layers['e2'] = x
    x = _encode_layer(x, base*4, reuse=reuse, scope=scope)
    enc_layers['e3'] = x
    x = _encode_layer(x, base*8, reuse=reuse, scope=scope)
    enc_layers['e4'] = x
    x = _encode_layer(x, base*8, reuse=reuse, scope=scope)
    enc_layers['e5'] = x
    x = _encode_layer(x, base*8, reuse=reuse, scope=scope)
    enc_layers['e6'] = x
    x = _encode_layer(x, base*8, reuse=reuse, scope=scope)
    enc_layers['e7'] = x
    x = _encode_layer(x, base*8, reuse=reuse, scope=scope)
    enc_layers['e8'] = x
    return enc_layers


def _decoder(inputs, base, enc_layers, ids, bank_size, reuse=False, scope=None):
    x = _decode_layer(inputs, base*8, sigma.shape(enc_layers['e7']), enc_layers['e7'], ids, bank_size,
                      drop=True,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base*8, sigma.shape(enc_layers['e6']), enc_layers['e6'], ids, bank_size,
                      drop=True,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base*8, sigma.shape(enc_layers['e5']), enc_layers['e5'], ids, bank_size,
                      drop=True,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base*8, sigma.shape(enc_layers['e4']), enc_layers['e4'], ids, bank_size,
                      drop=True,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base*4, sigma.shape(enc_layers['e3']), enc_layers['e3'], ids, bank_size,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base*2, sigma.shape(enc_layers['e2']), enc_layers['e2'], ids, bank_size,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base, sigma.shape(enc_layers['e1']), enc_layers['e1'], ids, bank_size,
                      reuse=reuse,
                      scope=scope)
    x = _decode_layer(x, base, sigma.shape(enc_layers['e0']), None, ids, bank_size,
                      reuse=reuse,
                      scope=scope)
    return layers.actives.tanh(x, reuse=reuse, scope=scope)
    

def _generator(inputs, embeddings, ids, embed_size, batch_size,
               reuse=False,
               scope=None):
    with sigma.scope('generator'):
        with sigma.defaults(reuse=reuse, scope=scope):
            enc_layers = _encoder(inputs)
            local_embeddings = layers.convs.embeddings(embeddings, ids=ids)
            local_embeddings = layers.base.reshape(local_embeddings, [batch_size, 1, 1, embed_size])
            embedded = layers.merge.concat([enc_layers['e8'], local_embeddings])
            dec = _decoder(embedded, 32, enc_layers, ids, reuse=reuse, scope=scope)
    return enc_layers, dec


def _discriminator(inputs, reuse=False, scope=None):
    with sigma.scope('discriminator'):
        with sigma.defaults(reuse=reuse, scope=scope):
            x = layers.convs.conv2d(inputs, act='leaky_relu')

            x = layers.convs.conv2d(x)
            x = layers.norms.batch_norm(x, act='leaky_relu')

            x = layers.convs.conv2d(x)
            x = layers.norms.batch_norm(x, act='leaky_relu')

            x = layers.convs.conv2d(x)
            x = layers.norms.batch_norm(x, act='leaky_relu')

            fc1 = layers.base.flatten(x)
            fc1 = layers.convs.dense(fc1)

            fc2 = layers.base.flatten(x)
            fc2 = layers.convs.dense(fc2)

            return layers.actives.sigmoid(fc1), fc1, fc2


def unet(input_shape, reuse=False, scope=None):
    inputs = layers.base.input_spec(input_shape, sigma.float32, reuse=reuse, scope=scope)
    