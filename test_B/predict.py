from keras.models import load_model
import cv2

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=600)])

image = cv2.imread("images/cat.jpg")
output = image.copy()
image = cv2.resize(image, (32, 32))

image = image.astype("float") / 255.0

image = image.flatten()
image = image.reshape((1, image.shape[0]))

model = load_model("model.h5")
preds = model.predict(image)
print(preds)
