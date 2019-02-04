import tensorflow as tf
from Tpu.Tpu_utils.gan_layers import *

class Discriminator:
    def __init__(self, batch_size,input_shape, rows, cols, channels, tpu=False):
        if tpu:
            self.dtype = tf.float16
        else:
            self.dtypt = tf.float32

        self.input_shape = input_shape
        self.output_width = rows
        self.output_height = cols
        self.n_classes = channels
        self.batch_size = batch_size

    def get_discriminator(self, images, getter, reuse=None):
        with tf.variable_scope('dis', reuse=reuse, custom_getter=getter):
            # conv1

            conv1 = conv_2d(images, 64, scope="conv1")

            # leakly ReLu
            h1 = leaky_relu(conv1)

            # conv2
            conv2 = conv_2d(h1, 128, scope="conv2")

            # batch norm
            norm2 = batch_norm(conv2, scope="batch_norm2", is_training=True)

            # leaky ReLU
            h2 = leaky_relu(norm2)

            # conv3
            conv3 = conv_2d(h2, 256, scope="conv3")

            # batch norm
            norm3 = batch_norm(conv3, scope="batch_norm3", is_training=True)

            # leaky ReLU
            h3 = leaky_relu(norm3)

            # conv4
            conv4 = conv_2d(h3, 512, scope="conv4")

            # batch norm
            norm4 = batch_norm(conv4, scope="batch_norm4", is_training=True)

            # leaky ReLU
            h4 = leaky_relu(norm4)

            # reshape
            h4_reshape = tf.reshape(h4, [self.batch_size, -1])

            # source logits
            source_logits = fc(h4_reshape, 1, scope="source_logits")

            # class logits
            class_logits = fc(
                h4_reshape, self.n_classes, scope="class_logits")

            return source_logits, class_logits

    def get_input_tensor(self):
        return tf.placeholder(self.dtypt, shape=[None, self.input_shape])
