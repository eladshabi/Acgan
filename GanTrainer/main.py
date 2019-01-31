from GanTrainer import GanTrainer
from GAN import ACGAN
from GanUtils import *
import sys
import os


if __name__ == "__main__":

    # Get all the parameters for the training.
    training_time = int(sys.argv[1])
    tpu = bool(sys.argv[2])


    root_path = os.path.dirname(os.path.abspath(__file__))
    batches = [128, 512, 1024]
    data_path = os.path.join(root_path,'data/')
    

    for batch in batches:
        folder_name = os.path.join(root_path, str(batch))
        os.makedirs(folder_name)

        # Create the Ac_gan model
        ac_gan = ACGAN(28, 28, 1, 10, 100,tpu)

        # Create the Gan trainer object and set the data loader function with the data folder path.

        gan_trainer = GanTrainer(ac_gan, folder_name, load_data,data_path)

        # Train the model for specific time and custom batch size
        gan_trainer.train_gan_by_time(training_time, batch,tpu)






