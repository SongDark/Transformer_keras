# encoding=utf-8
'''
@author: SongDark
@brief: transformer in keras
'''

import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from attention import *


class MyMaxPool(Layer):
    def __init__(self, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MyMaxPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x)!=K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0,2,1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.max(x, axis=self.axis, keepdims=False)
        else:
            return K.max(x, axis=self.axis, keepdims=False)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)


class MyMeanPool(Layer):
    def __init__(self, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MyMeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x)!=K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0,2,1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)


class Transformer(Layer):
    def __init__(self, model_dim=64, n_heads=2, encoder_stack=2, decoder_stack=2,
                 feed_forward_size=64, dropout_rate=0.2, **kwargs):
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
        self.supports_masking = True 
        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_encoder = self.add_weight(
            shape=(int(input_shape[0][-1]), self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_encoder'
        )
        self.weights_decoder = self.add_weight(
            shape=(int(input_shape[0][-1]), self._model_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_decoder'
        )
        super(Transformer, self).build(input_shape)

    def encode(self, inputs):
        # x.shape = [batch_size, seq_len, data_dim]

        # Embedding
        emb = tf.einsum("ntd,dk->ntk", inputs, self.weights_encoder)
        emb *= self._model_dim ** 0.5

        # Position Encoding
        emb += PositionEncoding(self._model_dim)(emb)
        # Dropout
        emb = K.dropout(emb, self._dropout_rate)

        # tiled blocks
        for i in range(self._encoder_stack):
            # Multi-Head Attention, Add & Norm
            attention = MultiheadAttention(self._n_heads, self._model_dim // self._n_heads)
            emb = attention([emb, emb, emb])

            # position-wise feedforward, Add & Norm
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            emb = ff(emb)

        return emb

    def decode(self, inputs):
        decoder_inputs, encoder_memory = inputs

        # Embedding
        emb = tf.einsum("ntd,dk->ntk", decoder_inputs, self.weights_decoder)
        emb *= self._model_dim ** 0.5

        # Position Encoding
        emb += PositionEncoding(self._model_dim)(emb)
        # Dropout
        emb = K.dropout(emb, self._dropout_rate)

        # Tiled Blocks
        for i in range(self._decoder_stack):
            # Self-Attention, Add & Norm
            masked_attention = MultiheadAttention(self._n_heads, self._model_dim // self._n_heads, future=True)
            emb = masked_attention([emb, emb, emb])

            # Multi-Head Attention, Add & Norm
            attention = MultiheadAttention(self._n_heads, self._model_dim // self._n_heads)
            emb = attention([emb, encoder_memory, encoder_memory])

            # position-wise feedforward, Add & Norm
            ff = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
            emb = ff(emb)

        return emb

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_encodings = self.encode(encoder_inputs)
        decoder_outputs = self.decode([decoder_inputs, encoder_encodings])
        return decoder_outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self._model_dim)

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        return mask 
    