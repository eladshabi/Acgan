# from Tpu.GanV3.gan import ACGAN
# from Tpu.GanV3.utils import show_all_variables
# import tensorflow as tf

from gan import ACGAN
from utils import show_all_variables
import tensorflow as tf

with tf.Session() as sess:

    #gan = ACGAN(sess,2,128,100,"mnist",'saved_model/','results/','logs', tpu=True)
    gan = ACGAN(sess, 1, 128, 100, "cifar10", 'saved_model/', 'results/', 'logs', tpu=True)

    # build graph
    gan.build_model()

    # show network architecture
    show_all_variables()

    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")



