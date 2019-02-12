# from Tpu.GanV3.gan import ACGAN
# from Tpu.GanV3.utils import show_all_variables

#from Tpu.GanV3v3.gan import ACGAN

from gan import ACGAN
from utils import show_all_variables

import tensorflow as tf
import sys
import os


def open_folders(mixed):

    if mixed:
        models = 'saved_model/Mixed/'
        results = 'results/Mixed/'
        logs = 'logs/Mixed/'

    else:
        models = 'saved_model/FP/'
        results = 'results/FP/'
        logs = 'logs/FP/'

    if not os.path.exists(models):
        os.makedirs(models)

    if not os.path.exists(results):
        os.makedirs(results)

    if not os.path.exists(logs):
        os.makedirs(logs)

    return models, results, logs


if __name__ == "__main__":

    # Get all the parameters for the training.

    data_set = sys.argv[1]
    training_time = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    tpu = bool(int(sys.argv[4]))

    model_file, results_file, logs_file = open_folders(tpu)

    with tf.Session() as sess:

        gan = ACGAN(sess, batch_size, batch_size, 100, data_set, model_file, results_file, logs_file, tpu=tpu)

        # build graph
        gan.build_model()

        # launch the graph in a session
        gan.train_by_time(training_time)
        print(" [*] Training finished!")



