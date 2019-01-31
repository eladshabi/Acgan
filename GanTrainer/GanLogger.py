import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from datetime import datetime
import logging


class Logger:
    def __init__(self, root_path):

        self.root_folder_path = root_path

        self.info_folder_path = os.path.join(self.root_folder_path, 'info')

        self.losses_folder_path = os.path.join(self.root_folder_path, 'losses')

        self.images_folder_path = os.path.join(self.root_folder_path, 'images')

        self.model_folder_path = os.path.join(self.root_folder_path, 'saved_model')

        if not os.path.exists(self.info_folder_path):
            os.makedirs(self.info_folder_path)

        if not os.path.exists(self.losses_folder_path):
            os.makedirs(self.losses_folder_path)

        if not os.path.exists(self.images_folder_path):
            os.makedirs(self.images_folder_path)

        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        logging.basicConfig(filename=os.path.join(self.info_folder_path, '.log'), level=logging.DEBUG)

    def write_warn_to_log(self, msg):
            logging.warning(msg)

    def write_info_to_log(self, msg):
            logging.info(msg)

    def write_debug_to_log(self,msg):
            logging.debug(msg)

    def write_losses(self, d_loss, g_loss):

        loss_file_path = os.path.join(self.losses_folder_path,'Loss.csv')

        if not os.path.exists(loss_file_path):
            with open(os.path.join(loss_file_path),'w') as writer_file:
                headers = ['Time', 'dis_loss', 'gen_loss']
                writer = csv.DictWriter(writer_file, headers)
                writer.writeheader()
                writer.writerow({'Time': datetime.now(), 'dis_loss': d_loss[0], 'gen_loss': g_loss[0]})
        else:
            with open(os.path.join(loss_file_path), 'a') as writer_file:
                headers = ['Time', 'dis_loss', 'gen_loss']
                writer = csv.DictWriter(writer_file, headers)
                writer.writerow({'Time': datetime.now(), 'dis_loss': d_loss[0], 'gen_loss': g_loss[0]})

    def save_images(self, generator, tpu):
        folder_name = str(datetime.now())
        folder_path = os.path.join(self.images_folder_path, folder_name)
        os.makedirs(folder_path)
        for class_id in range(10):
            os.makedirs(os.path.join(folder_path, str(class_id)))
            for i in range(10):
                noise = np.random.normal(0, 1, (1, 100))
                label = np.array([class_id]).astype(np.int32)

                if tpu:
                    noise = noise.astype(np.float16)
                else:
                    noise = noise.astype(np.float32)

                img = generator.predict([noise, label])
                img = img.astype(np.float32)
                plt.imshow(img.reshape(28, 28))
                plt.savefig(os.path.join(folder_path, str(class_id), str(i+1) + '.jpg'))
        plt.close()

    def save_model(self, generator, discriminator):
        def save(model, model_name):
            model_path = self.model_folder_path+"/temp_%s.json" % model_name
            weights_path = self.model_folder_path+"/temp_%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            with open(options['file_arch'], 'w') as f:
                f.write(json_string)
                model.save_weights(options['file_weight'])

            os.rename(model_path, self.model_folder_path+"/%s.json" % model_name)
            os.rename(weights_path, self.model_folder_path+"/%s_weights.hdf5" % model_name)

        save(generator, "generator")
        save(discriminator, "discriminator")
        self.write_info_to_log("model saved at: " + str(datetime.now()))



