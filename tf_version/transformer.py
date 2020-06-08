'''
@author: SongDark
@brief: tensorflow transformer
@ref: https://github.com/Kyubyong/transformer
'''

import tensorflow as tf
from attention import position_embedding, multihead_attention, ff

class Transformer():
    def __init__(self,
                 n_heads=[2, 2, 2],
                 n_blocks=[2, 2],
                 d_ff=64,
                 d_out=1,
                 d_model=64,
                 dropout_rate=0.0,
                 name=None):
        '''
            n_heads: list of 3 integers, [encoder_n, decoder_n1, decoder_n2]
            n_blocks: list of 2 integers, [encoder_block_n, decoder_block_n]
        '''
        self.name = name or "Transformer"
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.d_ff = d_ff
        self.d_out = d_out
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def encode(self, x, maxlen=None, is_training=True):
        '''
        Encoder
            x: 3d tensor. [N, T, d_in]
            maxlen: integer. must be >= T
            is_training: boolean.
            reuse: boolean.
        Returns:
            enc: # [N, T, d_model]
        '''
        with tf.variable_scope(self.name + "_encoder", reuse=tf.AUTO_REUSE):
            maxlen = maxlen or x.get_shape().as_list()[1]
            d_in = x.get_shape().as_list()[-1]

            # embedding, [N,T,d_data] -> [N,T,d_model]
            weights = tf.get_variable('weights',
                                      [d_in, self.d_model], tf.float32,
                                      tf.random_normal_initializer(stddev=0.02))
            x = tf.einsum("ntd,dk->ntk", x, weights)  # [N, T, d_model]
            enc = x * self.d_model ** 0.5  # scale

            # position encoding
            enc += position_embedding(enc, maxlen)  # [N, T, d_model]
            enc = tf.layers.dropout(enc, self.dropout_rate, training=is_training)

            # tiled blocks
            for i in range(self.n_blocks[0]):
                with tf.variable_scope("block_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc, att = multihead_attention(enc, enc, enc,
                                                   self.n_heads[0],
                                                   self.dropout_rate,
                                                   is_training=is_training,
                                                   causality=False)
                    # position-wise feedforward
                    enc = ff(enc, num_units=[self.d_ff, self.d_model])
        return enc, att

    def decode(self, y, memory, maxlen=None, is_training=True):
        '''
        Decoder
            y: 3d tensor. [N, T, d_in]
            memory: 3d tensor. [N, T, d_model]
        Returns:
        '''
        with tf.variable_scope(self.name + "_decoder", reuse=tf.AUTO_REUSE):
            maxlen = maxlen or y.get_shape().as_list()[1]
            d_in = y.get_shape().as_list()[-1]

            # embedding, [N,T,d_data] -> [N,T,d_model]
            weights = tf.get_variable('weights',
                                      [d_in, self.d_model], tf.float32,
                                      tf.random_normal_initializer(stddev=0.02))
            y = tf.einsum("ntd,dk->ntk", y, weights)  # [N, T, d_model]
            dec = y * self.d_model ** 0.5  # scale

            # position encoding
            dec += position_embedding(dec, maxlen)
            dec = tf.layers.dropout(dec, self.dropout_rate, training=is_training)

            # blocks
            for i in range(self.n_blocks[1]):
                with tf.variable_scope("block_{}".format(i), reuse=tf.AUTO_REUSE):
                    # masked self-attention
                    dec, _ = multihead_attention(dec, dec, dec,
                                                 self.n_heads[1],
                                                 self.dropout_rate,
                                                 is_training=is_training,
                                                 causality=True,
                                                 name='masked_selfattention')
                    # vanilla attention
                    dec, _ = multihead_attention(dec, memory, memory,
                                                 self.n_heads[2],
                                                 self.dropout_rate,
                                                 is_training=is_training,
                                                 causality=False,
                                                 name='vanilla_attention')
                    # ff
                    dec = ff(dec, num_units=[self.d_ff, self.d_model])  # [N, T, d_model]

            # output linear, convert [N, T, d_model] -> [N, T, d_out]
            weights = tf.get_variable('weights_out',
                                      [self.d_model, self.d_out], tf.float32,
                                      tf.random_normal_initializer(stddev=0.02))
            logits = tf.einsum('ntd,dk->ntk', dec, weights)  # [N, T, d_out]

        return logits

    @property
    def vars(self):
        encoder_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + "_encoder")
        decoder_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name + "_decoder")
        return encoder_collection + decoder_collection