
#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime, timedelta




from ops import *
from utils import *

# from Tpu.GanV3v3.ops import *
# from Tpu.GanV3v3.utils import *

from tensorflow.contrib.mixed_precision import FixedLossScaleManager,LossScaleOptimizer
from tensorflow.contrib.layers import fully_connected as fc


class ACGAN(object):
    model_name = "ACGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, tpu=False):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size

        if tpu:
            self.dtype = tf.float16
            self.nptype = np.float16
        else:
            self.dtype = tf.float32
            self.nptype = np.float32

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist' or dataset_name=='quick_draw' or dataset_name=="cifar10":
            # parameters
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = 10         # dimension of code-vector (label)
            self.c_dim = 3

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # code
            self.len_discrete_code = 10  # categorical distribution (i.e. label)
            self.len_continuous_code = 2  # gaussian distribution (e.g. rotation, thickness)

            # load mnist
            #self.data_X, self.data_y = load_mnist(self.dataset_name)

            # load quick draw
            #self.data_X, self.data_y = load_quick_draw(self.dataset_name, tpu)
            self.data_X, self.data_y = load_cifar10(tpu)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def classifier(self, x, is_training=True, reuse=False):

        with tf.variable_scope("classifier", reuse=reuse, custom_getter=float32_variable_storage_getter):

            net = fc(x, 128, scope='c_fc1', activation_fn=None)

            # Batch normalization should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            net = bn(net, is_training=is_training, scope='c_bn1')

            # Leveraging the tensors core for fully connected weight.
            net = tf.cast(net, tf.float16)
            net = tf.nn.leaky_relu(net, alpha=0.2)
            out_logit = fc(net, self.y_dim, scope='c_fc2', activation_fn=None)

            # Softmax should should be calculate as float32
            out_logit = tf.cast(out_logit, tf.float32)
            out = tf.nn.softmax(out_logit)

            return out, out_logit

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse, custom_getter=float32_variable_storage_getter):

            # Cast the input to float16
            x = tf.cast(x, tf.float16)

            net = conv2d(x, 64, 4, 4, 2, 2, name='d_conv1', data_type=self.dtype)
            net = tf.nn.leaky_relu(net, alpha=0.2)

            net = conv2d(net, 128, 4, 4, 2, 2, name='d_conv2', data_type=self.dtype)

            # Batch normalization should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            net = bn(net, is_training=is_training, scope='d_bn2')

            # Leveraging the tensors core for fully connected weight.
            net = tf.cast(net, tf.float16)
            net = tf.nn.leaky_relu(net, alpha=0.2)
            net = tf.reshape(net, [self.batch_size, -1])
            net = fc(net, 1024, scope='d_fc3', activation_fn=None)

            # Batch normalization should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            net = bn(net, is_training=is_training, scope='d_bn3')

            # Leveraging the tensors core for fully connected weight.
            net = tf.cast(net, tf.float16)
            net = tf.nn.leaky_relu(net, alpha=0.2)
            out_logit = fc(net, 1, scope='d_fc4', activation_fn=None)

            # Sigmoid should be calculated as type of float32
            out_logit = tf.cast(out_logit, tf.float32)
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, y, is_training=True, reuse=False):

        with tf.variable_scope("generator", reuse=reuse, custom_getter=float32_variable_storage_getter):

            # merge noise and code
            z = concat([z, y], 1)
            net = fc(z, 1024, scope='g_fc1', activation_fn=None)

            # Batch normalization should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            net = bn(net, is_training=is_training, scope='g_bn1')

            # Leveraging the tensors core for fully connected weight.
            net = tf.cast(net, tf.float16)
            net = tf.nn.relu(net)
            net = fc(net, 128 * 8 * 8, scope='g_fc2', activation_fn=None)

            # Batch normalization should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            net = bn(net, is_training=is_training, scope='g_bn2')

            # Leveraging the tensors core
            net = tf.cast(net, tf.float16)
            net = tf.nn.relu(net)
            net = tf.reshape(net, [self.batch_size, 8, 8, 128])
            net = deconv2d(net, [self.batch_size, 16, 16, 64], 4, 4, 2, 2, name='g_dc3', data_type=self.dtype)

            # Batch normalization should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            net = bn(net, is_training=is_training, scope='g_bn3')

            # Leveraging the tensors core
            net = tf.cast(net, tf.float16)
            net = tf.nn.relu(net)
            net = deconv2d(net, [self.batch_size, 32, 32, 3], 4, 4, 2, 2, name='g_dc4', data_type=self.dtype)

            # Sigmoid should be calculated as type of float32
            net = tf.cast(net, tf.float32)
            out = tf.nn.sigmoid(net)

            return out

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(self.dtype, [bs] + image_dims, name='real_images')

        # labels
        self.y = tf.placeholder(self.dtype, [bs, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(self.dtype, [bs, self.z_dim], name='z')

        """ Loss Function """
        ## 1. GAN Loss
        # output of D for real images
        D_real, D_real_logits, input4classifier_real = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, input4classifier_fake = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = tf.add(d_loss_real, d_loss_fake)

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        ## 2. Information Loss
        code_fake, code_logit_fake = self.classifier(input4classifier_fake, is_training=True, reuse=False)
        code_real, code_logit_real = self.classifier(input4classifier_real, is_training=True, reuse=True)

        # For real samples
        q_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=code_logit_real, labels=self.y))# Check for label dtype

        # For fake samples
        q_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=code_logit_fake, labels=self.y))

        # get information loss
        self.q_loss = tf.add(q_fake_loss, q_real_loss)

        """ Training """
        # divide trainable variables into a group for D and a group for G

        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        q_vars = [var for var in t_vars if ('d_' in var.name) or ('c_' in var.name) or ('g_' in var.name)]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate , beta1=self.beta1)
            self.q_optim = tf.train.AdamOptimizer(self.learning_rate , beta1=self.beta1)

            scale = 128

            self.loss_scale_manager_D = FixedLossScaleManager(scale)
            self.loss_scale_manager_G = FixedLossScaleManager(scale)
            self.loss_scale_manager_Q = FixedLossScaleManager(scale)

            print(3)

            self.loss_scale_optimizer_D = LossScaleOptimizer(self.d_optim, self.loss_scale_manager_D)
            self.loss_scale_optimizer_G = LossScaleOptimizer(self.g_optim, self.loss_scale_manager_G)
            self.loss_scale_optimizer_Q = LossScaleOptimizer(self.q_optim, self.loss_scale_manager_Q)

            print(4)

            self.grads_variables_D = self.loss_scale_optimizer_D.compute_gradients(self.d_loss,d_vars)
            self.grads_variables_G = self.loss_scale_optimizer_G.compute_gradients(self.g_loss,g_vars)
            self.grads_variables_Q = self.loss_scale_optimizer_Q.compute_gradients(self.q_loss,q_vars)

            self.q_grads = [(g,v) for (g,v) in self.grads_variables_Q if g is not None]

            self.training_step_op_D = self.loss_scale_optimizer_D.apply_gradients(self.grads_variables_D)
            self.training_step_op_G = self.loss_scale_optimizer_G.apply_gradients(self.grads_variables_G)
            self.training_step_op_Q = self.loss_scale_optimizer_Q.apply_gradients(self.q_grads)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        q_loss_sum = tf.summary.scalar("g_loss", self.q_loss)
        q_real_sum = tf.summary.scalar("q_real_loss", q_real_loss)
        q_fake_sum = tf.summary.scalar("q_fake_loss", q_fake_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.q_sum = tf.summary.merge([q_loss_sum, q_real_sum, q_fake_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_codes = self.data_y[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(self.nptype)

                # update D network
                _, summary_str, d_loss = self.sess.run([self.training_step_op_D, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.y: batch_codes,
                                                                  self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G & Q network
                _, summary_str_g, g_loss, _, summary_str_q, q_loss = self.sess.run(
                    [self.training_step_op_G, self.g_sum, self.g_loss, self.training_step_op_Q, self.q_sum, self.q_loss],
                    feed_dict={self.z: batch_z, self.y: batch_codes, self.inputs: batch_images})
                self.writer.add_summary(summary_str_g, counter)
                self.writer.add_summary(summary_str_q, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], './' + check_folder(
                        self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                        epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def train_by_time(self, train_time):
        def saver():
            self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            self.test_codes = self.data_y[0:self.batch_size]

            samples = self.sess.run(self.fake_images,
                                    feed_dict={self.z: self.sample_z, self.y: self.test_codes})
            tot_num_samples = min(self.sample_num, self.batch_size)
            manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
            manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
            save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], './' + check_folder(
                self.result_dir + '/' + self.model_dir) + '/' + self.model_name + 'final.png')


        def run_batch(idx):
            batch_images = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_codes = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]

            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(self.nptype)

            # update D network
            _, summary_str, d_loss = self.sess.run([self.training_step_op_D, self.d_sum, self.d_loss],
                                                   feed_dict={self.inputs: batch_images, self.y: batch_codes,
                                                              self.z: batch_z})

            # update G & Q network
            _, summary_str_g, g_loss, _, summary_str_q, q_loss = self.sess.run(
                [self.training_step_op_G, self.g_sum, self.g_loss, self.training_step_op_Q, self.q_sum, self.q_loss],
                feed_dict={self.z: batch_z, self.y: batch_codes, self.inputs: batch_images})

            print("Epoch: [%2d] [%4d/%4d] time: %s, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, idx, self.num_batches, str(datetime.now() - start_time), d_loss, g_loss))

        # initialize all variables
        tf.global_variables_initializer().run()

        epoch = 0
        batch_c = 0

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=train_time)

        while datetime.now() < end_time:

            if batch_c == self.num_batches:
                batch_c = 0
                epoch += 1

            run_batch(batch_c)
            batch_c += 1

        saver()
    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        """ random noise, random discrete code, fixed continuous code """
        y = np.random.choice(self.len_discrete_code, self.batch_size)
        y_one_hot = np.zeros((self.batch_size, self.y_dim))
        y_one_hot[np.arange(self.batch_size), y] = 1

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})

        save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.batch_size, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.batch_size, dtype=np.int64) + l
            y_one_hot = np.zeros((self.batch_size, self.y_dim))
            y_one_hot[np.arange(self.batch_size), y] = 1

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y_one_hot})
            save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim],
                        check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.len_discrete_code],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0