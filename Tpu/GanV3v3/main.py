# from Tpu.GanV3.gan import ACGAN
# from Tpu.GanV3.utils import show_all_variables

#from Tpu.GanV3v3.gan import ACGAN

from gan import ACGAN

import tensorflow as tf
import sys
import os


def open_folders(experiment_name, description):

    base_dir = os.path.join('/tmp', experiment_name )
    models = os.path.join(base_dir, 'saved_model/')
    results = os.path.join(base_dir, 'results/')
    logs =  os.path.join(base_dir, 'logs/')


    print("base dir :", base_dir)
    print("models dir :", models)
    print("results dir :", results)
    print("logs dir :", logs)


    if not os.path.exists(models):
        os.makedirs(models)

    if not os.path.exists(results):
        os.makedirs(results)

    if not os.path.exists(logs):
        os.makedirs(logs)

    with open(os.path.join(base_dir, 'description.txt'), 'w+') as f:
        f.write(description)

    return models, results, logs


if __name__ == "__main__":

    # Get all the parameters for the training.

    data_set = sys.argv[1]
    training_time = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    tpu = bool(int(sys.argv[4]))
    experiment_name = sys.argv[5]

    experiment_description = dict()
    experiment_description['training_time'] = training_time
    experiment_description['batch_size'] = batch_size
    experiment_description['tpu'] = tpu
    experiment_description['experiment_name'] = experiment_name
    experiment_description['data_set'] = data_set

    model_file, results_file, logs_file = open_folders(experiment_name, str(experiment_description))

    with tf.Session() as sess:

        gan = ACGAN(sess, batch_size, batch_size, 100, data_set, model_file, results_file, logs_file, tpu=tpu)

        # build graph
        gan.build_model()

        # launch the graph in a session
        gan.train_by_time(training_time)
        print(" [*] Training finished!")



