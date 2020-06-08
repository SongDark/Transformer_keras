'''
@author: SongDark
@brief: test on tensorflow networks
'''

import tensorflow as tf
import numpy as np
from attention import *
from transformer import *

def test():
    BATCH_SIZE = 64
    DATA_DIM = 6

    with tf.variable_scope('network'):
        transformer = Transformer(d_out=2, name='trans')

        source_seq = tf.placeholder(tf.float32, (BATCH_SIZE, None, DATA_DIM))
        memory, attention = transformer.encode(source_seq)
        network_out = transformer.decode(source_seq, memory)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        x = np.random.normal(0, 1, (BATCH_SIZE, 100, DATA_DIM))
        y = sess.run(network_out, feed_dict={source_seq: x})
        print(x.shape, '->', y.shape)

if __name__ == '__main__':
    test()

