import tensorflow as tf


class Generator:
    def __init__(self, letant_size, tpu=False):

        if tpu:
            self.dtype = tf.float16
        else:
            self.dtypt = tf.float32

        self.letant_size = letant_size

    def get_generator(self, z, getter, reuse=None):
        with tf.variable_scope('gen', reuse=reuse, custom_getter=getter):
            hidden1 = tf.layers.dense(inputs=z, units=128)
            # Leaky Relu
            alpha = 0.01
            hidden1 = tf.maximum(alpha * hidden1, hidden1)
            hidden2 = tf.layers.dense(inputs=hidden1, units=128)

            hidden2 = tf.maximum(alpha * hidden2, hidden2)
            output = tf.layers.dense(hidden2, units=784, activation=tf.nn.tanh)
            return output

    def get_input_tensor(self):
        return tf.placeholder(self.dtype, shape=[None, self.letant_size])


