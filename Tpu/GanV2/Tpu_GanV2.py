import numpy as np
import time
from tensorflow.contrib.mixed_precision import FixedLossScaleManager, LossScaleOptimizer
from tensorflow.examples.tutorials.mnist import input_data
from Tpu.GanV2.Generator import Generator
from Tpu.GanV2.Discriminator import Discriminator
from Tpu.GanV2.losses import *

from Tpu.GanV2.dtype_convert import float32_variable_storage_getter

mnist = input_data.read_data_sets("data/", one_hot=True)

#mnist = tf.keras.datasets.mnist


def get_data():
    return mnist.load_data()



def create_gan_model():

    gen = Generator(letant_size=100, tpu=True)
    dis = Discriminator(128, True)

    labels = tf.placeholder(tf.int32, shape=[None])
    real_images = dis.get_input_tensor()
    z = gen.get_input_tensor()

    G = gen.get_generator(z, labels)
    print("gen :",G)

    D_output_real, D_logits_real = dis.get_discriminator(G, labels)
    D_output_fake, D_logits_fake = dis.get_discriminator(G,labels,reuse=True)

    #############################
    #                           #
    #      Loss functions       #
    #                           #
    #############################

    D_loss , G_loss= loss(labels, D_output_real, D_logits_real, D_output_fake, D_logits_fake, G)

    # Get all the trainable variables
    tvars = tf.trainable_variables()

    d_vars = [var for var in tvars if 'dis' in var.name]
    g_vars = [var for var in tvars if 'gen' in var.name]

    # Standard Optimizers
    D_trainer = tf.train.AdamOptimizer(0.002, 0.5)
    G_trainer = tf.train.AdamOptimizer(0.002, 0.5)

    loss_scale_manager_D = FixedLossScaleManager(5000)
    loss_scale_manager_G = FixedLossScaleManager(5000)
    loss_scale_optimizer_D = LossScaleOptimizer(D_trainer, loss_scale_manager_D)
    loss_scale_optimizer_G = LossScaleOptimizer(G_trainer, loss_scale_manager_G)

    grads_variables_D = loss_scale_optimizer_D.compute_gradients(D_loss, d_vars)
    grads_variables_G = loss_scale_optimizer_G.compute_gradients(G_loss, g_vars)

    training_step_op_D = loss_scale_optimizer_D.apply_gradients(grads_variables_D)
    training_step_op_G = loss_scale_optimizer_D.apply_gradients(grads_variables_G)

    init = tf.global_variables_initializer()

    samples = []

    batch_size = 128
    epochs = 100

    saver = tf.train.Saver(var_list=g_vars)

    with tf.Session() as sess:

        sess.run(init)

        start = time.time()

        # Recall an epoch is an entire run through the training data
        for e in range(epochs):
            # // indicates classic division
            num_batches = mnist.train.num_examples // batch_size

            for i in range(num_batches):
                # Grab batch of images
                batch = mnist.train.next_batch(batch_size)

                # Get images, reshape and rescale to pass to D
                batch_images = batch[0].astype(np.float16).reshape((batch_size, 784))
                batch_images = batch_images * 2 - 1

                # Z (random latent noise data for Models)
                # -1 to 1 because of tanh activation
                batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))

                # Run optimizers, no need to save outputs, we won't use them
                _ = sess.run(training_step_op_D, feed_dict={real_images: batch_images, z: batch_z})
                _ = sess.run(training_step_op_G, feed_dict={z: batch_z})

            print("Currently on Epoch {} of {} total...".format(e + 1, epochs))

            # Sample from generator as we're training for viewing afterwards
            #sample_z = np.random.uniform(-1, 1, size=(1, 100))
            # gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: sample_z})
            #
            # samples.append(gen_sample)
            saver.save(sess, '/models/model.ckpt')

        end = time.time()
        print(end - start)


if __name__ == "__main__":
    create_gan_model()