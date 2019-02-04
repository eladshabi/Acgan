import tensorflow as tf


class Discriminator:
    def __init__(self, input_shape, tpu=False):

        if tpu:
            self.dtype = tf.float16
        else:
            self.dtypt = tf.float32

        self.input_shape = input_shape

    def get_discriminator(X, getter, reuse=None):
        with tf.variable_scope('dis', reuse=reuse, custom_getter=getter):
            hidden1 = tf.layers.dense(inputs=X, units=128)
            # Leaky Relu
            alpha = 0.01
            hidden1 = tf.maximum(alpha * hidden1, hidden1)

            hidden2 = tf.layers.dense(inputs=hidden1, units=128)
            hidden2 = tf.maximum(alpha * hidden2, hidden2)

            logits = tf.layers.dense(hidden2, units=1)
            output = tf.sigmoid(logits)

            return output, logits

    def get_input_tensor(self):
        return tf.placeholder(self.dtypt, shape=[None, self.input_shape])
