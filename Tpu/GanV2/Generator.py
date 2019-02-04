from Tpu.GanV2.gan_layers import *


class Generator:
    def __init__(self, batch_size, letant_size, rows, cols, channels, tpu=False):
        if tpu:
            self.dtype = tf.float16
        else:
            self.dtypt = tf.float32

        self.letant_size = letant_size

        self.output_width = rows
        self.output_height = cols
        self.n_classes = channels
        self.batch_size = batch_size

    def get_generator(self, z, labels, getter, reuse=None):
        with tf.variable_scope('gen', reuse=reuse, custom_getter=getter):
            labels_one_hot = tf.one_hot(labels, self.n_classes)

            # concat z and labels
            z_labels = tf.concat(1, [z, labels_one_hot])

            # project z and reshape
            oh, ow = self.output_height, self.output_width

            z_labels_ = fc(z_labels, 512 * oh / 16 * ow / 16, scope="project")
            z_labels_ = tf.reshape(z_labels_, [-1, oh / 16, ow / 16, 512])

            # batch norm
            norm0 = batch_norm(
                z_labels_, scope="batch_norm0", is_training=True)

            # ReLU
            h0 = tf.nn.relu(norm0)

            # conv1
            conv1 = conv2d_transpose(
                h0, [self.batch_size, oh / 8, ow / 8, 256],
                scope="conv_tranpose1")

            # batch norm
            norm1 = batch_norm(conv1, scope="batch_norm1", is_training=True)

            # ReLU
            h1 = tf.nn.relu(norm1)

            # conv2
            conv2 = conv2d_transpose(
                h1, [self.batch_size, oh / 4, ow / 4, 128],
                scope="conv_tranpose2")

            # batch norm
            norm2 = batch_norm(conv2, scope="batch_norm2", is_training=True)

            # ReLU
            h2 = tf.nn.relu(norm2)

            # conv3
            conv3 = conv2d_transpose(
                h2, [self.batch_size, oh / 2, ow / 2, 64], scope="conv_tranpose3")

            # batch norm
            norm3 = batch_norm(conv3, scope="batch_norm3", is_training=True)

            # ReLU
            h3 = tf.nn.relu(norm3)

            # conv4
            conv4 = conv2d_transpose(
                h3, [self.batch_size, oh, ow, self.input_channels],
                scope="conv_tranpose4")

            # tanh
            h4 = tf.nn.tanh(conv4)

        return h4

    def get_input_tensor(self):
        return tf.placeholder(self.dtype, shape=[None, self.letant_size])


