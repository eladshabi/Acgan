import os
import numpy as np
import random
import tensorflow as tf

def get_name(name):
    return dict(
        {
            'airplane': 0,
            'apple': 1,
            'bee': 2,
            'bird': 3,
            'book': 4,
            'clock': 5,
            'cow': 6,
            'dog': 7,
            'eye': 8,
            'fish': 9

        }).get(name)


def load_data(path,tpu=False):
    images = []
    labels = []

    for filename in os.listdir(path):

        c_num = get_name(filename.split('.')[0])

        if filename.endswith('.npy'):
            data = np.load(os.path.join(path, filename))
            for img in data:
                images.append(img.reshape(28, 28))
                labels.append(c_num)

    data = list(zip(images, labels))

    random.shuffle(data)

    images, labels = zip(*data)
    if tpu:
        #images = np.array(images).astype(np.float16)
        images = tf.image.convert_image_dtype(images,tf.float16)
    else:
        images = np.array(images).astype(np.float32)
    images = (images - 127.5) / 127.5
    images = np.expand_dims(images, axis=3)
    labels = np.array(labels)
    labels = labels.reshape(-1, 1)
    return images, labels

