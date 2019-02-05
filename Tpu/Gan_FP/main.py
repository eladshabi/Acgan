from GAN import ACGAN
from utils import show_all_variables
import tensorflow as tf

with tf.Session() as sess:

    gan = ACGAN(sess, 2, 128, 100, "quick_draw", 'saved_model/', 'results/', 'logs')

    # build graph
    gan.build_model()

    # show network architecture
    show_all_variables()

    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")