# encoding=utf-8
'''
@author: SongDark
@brief: transformer in keras
'''

import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
from attention import *

class Transformer(Layer):
    def __init__(self, model_dim=64, n_heads=2, encoder_stack=2, decoder_stack=2,
                 feed_forward_size=64, dropout_rate=0.2, **kwargs):
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._encoder_stack = encoder_stack
        self._decoder_stack = decoder_stack
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate
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
        # self.weights_output = self.add_weight(
        #     shape=(int(input_shape[0][-1]), self._model_dim),
        #     initializer='glorot_uniform',
        #     trainable=True,
        #     name='weights_out'
        # )
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










