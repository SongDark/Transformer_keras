# encoding:utf-8
'''
@author: SongDark
@brief: multi-head attention
#ref: https://github.com/Kyubyong/transformer
'''

import tensorflow as tf


def mask(x, quries=None, keys=None, type=None):
    '''
    mask paddings on keys or queries to inputs
        x: 3d tensor. [N, T_q, T_k]
        quries: 3d tensor. [N, T_q, d]
        keys: 3d tensor. [N, T_k, d]
    '''
    padding_num = -2 ** 32 + 1
    if type in ('K', 'k', 'key', 'keys'):
        # generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # [N, T_k]
        masks = tf.expand_dims(masks, 1)  # [N, 1, T_k]
        masks = tf.tile(masks, [1, tf.shape(quries)[1], 1])  # [N, T_q, T_k]
        # apply masks to inputs
        paddings = tf.ones_like(x) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, x)
    elif type in ('Q', 'q', 'query', 'queries'):
        # generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(quries), axis=-1))  # [N, T_q]
        masks = tf.expand_dims(masks, -1)  # [N, T_q, 1]
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # [N, T_q, T_k]
        # apply masks to inputs
        outputs = x * masks
    elif type in ('f', 'future', 'right'):
        diag_vals = tf.ones_like(x[0, :, :])  # [T_q, T_k]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # [T_q, T_k]
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(x)[0], 1, 1])  # [N, T_q, T_k]
        # apply masks
        paddings = tf.ones_like(masks) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, x)

    return outputs


def layer_norm(x, epsilon=1e-8, name='ln'):
    '''
    x: nd tensor. [N, ...]
    epsilon: float.
    returns:
        tensor with the same shape of x.
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x_shape = x.get_shape()
        p_shape = x_shape[-1:]

        mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
        beta = tf.get_variable('beta', p_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', p_shape, initializer=tf.ones_initializer())
        norm = (x - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * norm + beta

    return outputs


def scaled_dot_product_attention(Q, K, V, causality=False, dropout_rate=0.0, is_training=True,
                                 name='scaled_dot_product_attention'):
    '''
        Q: 3d tensor. [N, T_q, d_k]
        K: 3d tensor. [N, T_k, d_k]
        V: 3d tensor. [N, T_k, d_v]
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product, QK^T
        outputs = tf.matmul(Q, tf.transpose(K, (0, 2, 1)))  # [N, T_q, T_k]

        # scale
        outputs /= (d_k ** 0.5)

        # key masking
        outputs = mask(outputs, Q, K, type='key')

        # future blind masking
        if causality:
            outputs = mask(outputs, type='future')

        # softmax
        outputs = tf.nn.softmax(outputs)  # [N, T_q, T_k]
        # attention = tf.transpose(outputs, [0,2,1]) # [N, T_k, T_q]
        attention = outputs
        # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        outputs = mask(outputs, Q, K, type='query')

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        # weighted sum
        outputs = tf.matmul(outputs, V)  # [N, T_q, d_v]

    return outputs, attention


def position_embedding(x, mode='sum', pos_dim=None):
    batch_size, T, d = tf.shape(x)[0], tf.shape(x)[1], x.get_shape().as_list()[2]
    pos_dim = pos_dim or d

    positions = tf.range(tf.to_float(T), dtype=tf.float32)
    positions = tf.expand_dims(positions, 1)  # [T, 1]

    # [pos_dim/2, ]
    dimension_even = 1.0 / tf.pow(10000.0, 2 * tf.range(0, pos_dim, delta=2, dtype=tf.float32) / pos_dim)
    dimension_odd = 1.0 / tf.pow(10000.0, 2 * tf.range(1, pos_dim, delta=2, dtype=tf.float32) / pos_dim)

    if pos_dim % 2 == 1:
        dimension_odd = tf.concat([dimension_odd, tf.zeros(shape=(1,))], 0)

    position_even = tf.sin(tf.matmul(positions, tf.expand_dims(dimension_even, 0)))  # [T, pos_dim/2]
    position_odd = tf.cos(tf.matmul(positions, tf.expand_dims(dimension_odd, 0)))  # [T, pos_dim/2]

    position_emb = tf.concat([tf.expand_dims(position_even, -1),
                              tf.expand_dims(position_odd, -1)], -1)
    position_emb = tf.reshape(position_emb, (1, T, -1))

    if pos_dim % 2 == 1:
        position_emb = position_emb[:, :, :-1]
    position_emb = position_emb + tf.zeros(shape=(batch_size, T, pos_dim))

    # if mode=='sum':
    #     return position_emb + x
    # elif mode == 'concat':
    #     return tf.concat([x, position_emb], -1)
    return position_emb


def ff(x, num_units, reuse=False, name=None):
    '''
    position-wise feed forward network (FFN), with 2 linear transformations, a ReLU in between.
        x: 3d tensor. [N, T, d]
        num_units: list of 2 integers. The second must be d.
    Returns:
        outputs: 3d tensor. [N, T, d]
    '''
    with tf.variable_scope(name or "FF", reuse=reuse):
        # inner layer
        outputs = tf.layers.dense(x, num_units[0], activation=tf.nn.relu)  # [N, T, h1]
        # outer layer
        outputs = tf.layers.dense(outputs, num_units[1])  # [N, T, d]
        # residual connection
        outputs += x  # [N, T, d]
        # norm
        outputs = layer_norm(outputs)
    return outputs


def multihead_attention(Q, K, V, n_head,
                        dropout_rate=0.0,
                        causality=False,
                        is_training=True, reuse=False,
                        name=None):
    '''
    Multi-Head Attention
        Q: 3d tensor. Query, [N, T_q, d_model],
            d_model must be evenly divisible by n_head, e.g. d_model=8, n_head=2
        K: 3d tensor. Keys, [N, T_k, d_model]
        V: 3d tensor. Values, [N, T_k, d_model]
            Keys must have the same shape as Values.
        n_head: integer. Number of head.
        dropout_rate: float. Rate to drop.
        causality: boolean. Mask future if True.
        is_training: boolean.
    Returns:
        outputs: 3d tensor. [N, T_q, d_model], with the same shape as Q.
    '''
    d_model = Q.get_shape().as_list()[-1]

    with tf.variable_scope(name or "multihead_attention", reuse=reuse):
        # Projection
        Q = tf.layers.dense(Q, d_model)  # [N, T_q, d_model]
        K = tf.layers.dense(K, d_model)  # [N, T_k, d_model]
        V = tf.layers.dense(V, d_model)  # [N, T_k, d_model]

        # split and concat
        Q_ = tf.concat(tf.split(Q, n_head, axis=2), axis=0)  # [h*N, T_q, d_model / h]
        K_ = tf.concat(tf.split(K, n_head, axis=2), axis=0)  # [h*N, T_k, d_model / h]
        V_ = tf.concat(tf.split(V, n_head, axis=2), axis=0)  # [h*N, T_k, d_model / h]

        # Attention
        outputs, attention = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate,
                                                          is_training)  # [h*N, T_q, d_model / h]
        outputs = tf.concat(tf.split(outputs, n_head, axis=0), axis=2)  # [N, T_q, d_model]
        # Residual conncetion
        outputs += Q  # [N, T_q, d_model]

        # Norm
        outputs = layer_norm(outputs)  # [N, T_q, d_model]

    return outputs, attention