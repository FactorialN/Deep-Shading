import tensorflow as tf


def AutoEncoderLikeCNN(x, reuse):
    with tf.variable_scope("model", reuse=reuse):
        n = int(x.shape[1])
        u = int(x.shape[3])
        x_i = x
        namei = 0
        sv = []
        while n > 128:
            sv.append(x_i)
            l_relu = tf.nn.leaky_relu(tf.layers.conv2d(x_i, u * 2, 4, padding='same', kernel_initializer=tf.truncated_normal_initializer(mean=0.01), name='conv' + str(namei), trainable=True), alpha=0.01)
            x_i = tf.layers.average_pooling2d(l_relu, pool_size=2, strides=(2, 2))
            n /= 2
            u *= 2
            namei += 1
        x_i = tf.nn.leaky_relu(x_i, alpha=0.01)
        while n < int(x.shape[1]):
            namei -= 1
            us = tf.image.resize_bilinear(x_i, (int(n * 2), int(n * 2)))
            cct = tf.concat([us, sv[namei]], axis=3)
            l_relu = tf.nn.leaky_relu(tf.layers.conv2d(cct, u, 4, padding='same', kernel_initializer=tf.truncated_normal_initializer(mean=0.01), name='convb' + str(namei), trainable=True), alpha=0.01)
            n *= 2
            u /= 2
            x_i = l_relu
        logits = tf.layers.dense(x_i, 1, name='linear', trainable=True)
    logits = (tf.nn.tanh(logits) + 1) / 2
    return logits

# can be extended