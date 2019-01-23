from GanTrainer import GanTrainer
from GAN import ACGAN
from GanUtils import *
import sys
import os


if __name__ == "__main__":

    # Get all the parameters for the training.
    training_time = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Create the Ac_gan model
    ac_gan = ACGAN(28, 28, 1, 10, 100)

    # Create the Gan trainer object and set the data loader function with the data folder path.
    gan_trainer = GanTrainer(ac_gan, root_path, load_data)

    # Train the model for specific time and custom batch size
    gan_trainer.train_gan_by_time(training_time, batch_size)





