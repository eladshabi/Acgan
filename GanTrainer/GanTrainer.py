import numpy as np
from datetime import datetime, timedelta
from GanLogger import Logger


class GanTrainer:
    def __init__(self, gan_model, root_path, data_loader):

        ###############
        #     GAN     #
        ###############

        self.combined = gan_model.combined
        self.discriminator = gan_model.discriminator
        self.generator = gan_model.generator

        self.gan = gan_model

        self.X_train = None
        self.y_train = None

        # Data loader:  needs to have load_data(path) function and has to normalize the data to [-1,1]
        # range as float32 np array
        # path: the Dataset dir
        # Return: (X_train, y_train)
        self.data_loader = data_loader

        # Path of the root directory
        self.root_path = root_path

        self.logger = Logger(root_path)

    def load_dataset(self):
        try:
            return self.data_loader(os.path.join(self.root_path ,'data/'))

        except:
            Emsg = 'Error occurred during loading the data'
            self.logger.write_warn_to_log(Emsg)

    def train_gan_by_time(self, time, batch_size=128):

        def train_discriminator(noise, sampled_labels, idx, imgs, valid_l, fake_l):
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])
            img_labels = self.y_train[idx]
            fake_labels = sampled_labels

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid_l, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake_l, fake_labels])
            dis_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            return dis_loss

        def run_batch(valid_l, fake_l, batch):
            # Select a random batch of images
            idx = np.random.randint(0, train_size, batch)
            imgs = self.X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch, self.gan.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.gan.num_of_classes, (batch, 1))

            #  Train the Discriminator

            dis_loss = train_discriminator(noise, sampled_labels, idx, imgs, valid_l, fake_l)

            # Train the generator
            gen_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            return dis_loss, gen_loss

        self.X_train, self.y_train = self.load_dataset()

        self.logger.write_info_to_log('Dataset loaded')

        train_size = self.X_train.shape[0]

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        batch_learned_in_epoch = 0
        epoch = 0

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=time)

        saving_step_time = 5
        saving_time = start_time + timedelta(minutes=saving_step_time)

        while datetime.now() < end_time:
            if batch_learned_in_epoch + batch_size < train_size:
                d_loss, g_loss = run_batch(valid, fake, batch_size)
                batch_learned_in_epoch += batch_size

            else:
                d_loss, g_loss = run_batch(valid, fake, train_size - batch_learned_in_epoch)
                self.logger.save_model(self.generator, self.discriminator)
                batch_learned_in_epoch = 0
                epoch += 1

            if datetime.now() > saving_time:
                self.logger.save_images(self.generator)
                saving_time = datetime.now() + timedelta(minutes=saving_step_time)

            self.logger.write_losses(d_loss, g_loss)












