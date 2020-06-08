# encoding:utf-8
'''
@author: SongDark
@brief: multi-head attention
'''

import keras.backend as K
import tensorflow as tf
from keras.layers import Layer

class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(shape=[int(input_shape[-1])], initializer='zero', name='beta')
        self.gamma = self.add_weight(shape=[int(input_shape[-1])], initializer='one', name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        norm = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * norm + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class ScaledDotProductAttention(Layer):
    def __init__(self, masking=True, future=False, dropout_rate=0.0, **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, quries=None, keys=None, mask_type=None):
        if mask_type in ('K', 'k', 'key', 'keys'):
            masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # [N, T_k]
            masks = tf.expand_dims(masks, 1)  # [N, 1, T_k]
            masks = tf.tile(masks, [1, tf.shape(quries)[1], 1])  # [N, T_q, T_k]
            paddings = tf.ones_like(inputs) * self._masking_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        elif mask_type in ('Q', 'q', 'query', 'queries'):
            masks = tf.sign(tf.reduce_sum(tf.abs(quries), axis=-1))  # [N, T_q]
            masks = tf.expand_dims(masks, -1)  # [N, T_q, 1]
            masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # [N, T_q, T_k]
            outputs = inputs * masks
        elif mask_type in ('f', 'future', 'right'):
            diag_vals = tf.ones_like(inputs[0, :, :])  # [T_q, T_k]
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # [T_q, T_k]
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # [N, T_q, T_k]
            paddings = tf.ones_like(masks) * self._masking_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

        return outputs

    def call(self, inputs):
        # inputs = [Q, K, V]
        assert len(inputs) == 3
        queries, keys, values = inputs

        d_k = queries.get_shape().as_list()[-1]

        # dot product, QK^T
        outputs = tf.matmul(queries, tf.transpose(keys, (0, 2, 1)))  # [N, T_q, T_k]

        # scale
        outputs /= (d_k ** 0.5)

        # key masking
        if self._masking:
            outputs = self.mask(outputs, queries, keys, mask_type='key')

        # future blind masking
        if self._future:
            outputs = self.mask(outputs, mask_type='future')

        # softmax
        outputs = tf.nn.softmax(outputs)  # [N, T_q, T_k]
        # attention = outputs

        # query masking
        outputs = self.mask(outputs, queries, keys, mask_type='query')

        # dropout
        outputs = K.dropout(outputs, self._dropout_rate)

        # weighted sum
        outputs = tf.matmul(outputs, values)  # [N, T_q, d_v]

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionEncoding(Layer):
    def __init__(self, pos_dim=None, **kwargs):
        self._pos_dim = pos_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        batch_size, T, d = tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape().as_list()[2]
        pos_dim = self._pos_dim or d

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

        return position_emb

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(int(input_shape[-1]), self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_inner'
        )
        self.bias_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name='bias_inner'
        )
        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_out'
        )
        self.bias_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name='bias_out'
        )
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bias_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bias_out
        outputs += inputs
        outputs = LayerNormalization()(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self._model_dim)


class MultiheadAttention(Layer):
    def __init__(self, n_heads, head_dim, dropout_rate=0.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiheadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(int(input_shape[0][-1]), self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries'
        )
        self._weights_keys = self.add_weight(
            shape=(int(input_shape[0][-1]), self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys'
        )
        self._weights_values = self.add_weight(
            shape=(int(input_shape[0][-1]), self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values'
        )
        super(MultiheadAttention, self).build(input_shape)

    def call(self, inputs):
        assert len(inputs) == 3
        queries, keys, values = inputs

        # Projection
        queries_p = K.dot(queries, self._weights_queries)
        keys_p = K.dot(keys, self._weights_keys)
        values_p = K.dot(values, self._weights_values)

        # split and concat
        queries_s = tf.concat(tf.split(queries_p, self._n_heads, axis=2), axis=0)
        keys_s = tf.concat(tf.split(keys_p, self._n_heads, axis=2), axis=0)
        values_s = tf.concat(tf.split(values_p, self._n_heads, axis=2), axis=0)

        # attention
        attention = ScaledDotProductAttention(self._masking, self._future, self._dropout_rate)
        att_out = attention([queries_s, keys_s, values_s])
        att_out = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)

        # outputs
        # outputs = att_out + queries
        outputs = LayerNormalization()(att_out + queries)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape




