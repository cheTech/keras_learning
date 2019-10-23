"""
using neural network

Author: cheTech
Github: https://github.com/cheTech/keras_learning

Docs:
https://www.reg.ru/blog/keras/
"""

from tensorflow.keras import models
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
import cv2
import pandas as pd
import numpy as np
import pickle

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=150)])

image_size = 784


def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width))
            for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    img_array = np.expand_dims(img_array, axis=0)
    output = img_array  # preprocess_input(img_array)
    return(output)

model = models.load_model('my_model.h5')

test_data = read_and_prep_images(['test_set/orig.jpg'])

preds = model.predict(test_data)
print(preds)
