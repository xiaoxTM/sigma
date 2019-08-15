import tensorflow as tf

idx = tf.constant(0, dtype=tf.int32)
iterations = 3

def body(idx, array):
    array = tf.cond(tf.equal(idx+1,iterations), lambda: array.write(idx, -1), lambda: array.write(idx, idx+1))
    #if idx + 1 == iterations:
    #    array = array.write(idx, -1)
    #else:
    #    array = array.write(idx, idx+1)
    return (idx+1, array)

array = tf.TensorArray(dtype=tf.int32,
                       size=iterations)
_, res = tf.while_loop(
        lambda idx, array: idx < iterations,
        body,
        loop_vars=[idx, array]
        )
x = res.stack()

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print(sess.run(x))
