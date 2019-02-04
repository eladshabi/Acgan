from Tpu.GanV2.gan_layers import *
from Tpu.GanV2.dtype_convert import float32_variable_storage_getter


class Generator:
    def __init__(self, batch_size=128, letant_size=100, rows=28, cols=28, channels=1, tpu=False):
        if tpu:
            self.dtype = tf.float16
        else:
            self.dtype = tf.float32

        self.letant_size = letant_size

        self.output_width = rows
        self.output_height = cols
        self.channels = channels
        self.batch_size = batch_size
        self.n_classes = 10

    def get_generator(self, z, labels, reuse=None):
        with tf.variable_scope('gen', reuse=reuse, custom_getter=float32_variable_storage_getter):

            # merge noise and code
            z = tf.concat([z, y], 1)

            net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 7, 7, 128])
            net = tf.nn.relu(
                bn(deconv2d(net, [self.batch_size, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))

            out = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

            return out
        #     labels_one_hot = tf.one_hot(labels, self.n_classes, dtype=tf.float16)
        #
        #     # concat z and labels
        #     z_labels = tf.concat([z, labels_one_hot],1)
        #
        #     # project z and reshape
        #     oh, ow = self.output_height, self.output_width
        #
        #     z_labels_ = fc(z_labels, 512 * oh / 7 * ow / 7, scope="project")
        #     print(z_labels)
        #
        #     print([-1, oh / 7, ow / 7, 512])
        #
        #     z_labels_ = tf.reshape(z_labels_, [-1, 4, 4, 512])
        #
        #     # batch norm
        #     norm0 = batch_norm(
        #         z_labels_, scope="batch_norm0", is_training=True)
        #
        #     # ReLU
        #     h0 = tf.nn.relu(norm0)
        #
        #     # conv1
        #     conv1 = conv2d_transpose(
        #         h0, [self.batch_size, 14, 14, 256],
        #         scope="conv_tranpose1")
        #
        #     # batch norm
        #     norm1 = batch_norm(conv1, scope="batch_norm1", is_training=True)
        #
        #     # ReLU
        #     h1 = tf.nn.relu(norm1)
        #
        #     # conv2
        #     conv2 = conv2d_transpose(
        #         h1, [self.batch_size, 14, 14, 128],
        #         scope="conv_tranpose2")
        #
        #     # batch norm
        #     norm2 = batch_norm(conv2, scope="batch_norm2", is_training=True)
        #
        #     # ReLU
        #     h2 = tf.nn.relu(norm2)
        #
        #     # conv3
        #     conv3 = conv2d_transpose(
        #         h2, [self.batch_size, 7, 7, 64], scope="conv_tranpose3")
        #
        #     # batch norm
        #     norm3 = batch_norm(conv3, scope="batch_norm3", is_training=True)
        #
        #     # ReLU
        #     h3 = tf.nn.relu(norm3)
        #
        #     # conv4
        #     conv4 = conv2d_transpose(
        #         h3, [self.batch_size, oh, ow, self.channels],
        #         scope="conv_tranpose4")
        #
        #     # tanh
        #     h4 = tf.nn.tanh(conv4)
        #
        # return h4

    def get_input_tensor(self):
        return tf.placeholder(self.dtype, shape=[None, self.letant_size])


