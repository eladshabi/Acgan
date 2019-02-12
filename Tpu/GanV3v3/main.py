# from Tpu.GanV3.gan import ACGAN
# from Tpu.GanV3.utils import show_all_variables

#from Tpu.GanV3v3.gan import ACGAN


from gan import ACGAN
from utils import show_all_variables

import tensorflow as tf
import sys
import os

if __name__ == "__main__":

    # Get all the parameters for the training.

    data_set = sys.argv[1]
    training_time = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    tpu = bool(sys.argv[4])

    if not os.path.exists('saved_model/'):
        os.makedirs('saved_model/')

    if not os.path.exists('results/'):
        os.makedirs('results/')

    if not os.path.exists('logs'):
        os.makedirs('logs')

    with tf.Session() as sess:

        gan = ACGAN(sess, batch_size, batch_size, 100, data_set, 'saved_model/', 'results/', 'logs', tpu=tpu)

        # build graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        # launch the graph in a session
        gan.train_by_time(training_time)
        print(" [*] Training finished!")



