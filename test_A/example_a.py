"""
train neural network

Author: cheTech
Github: https://github.com/cheTech/keras_learning

Docs:
https://www.reg.ru/blog/keras/
"""

from tensorflow.keras.datasets import fashion_mnist  # loading dataset
# для представления нейронной сети
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense  # модуль полносвязной нейронной сети
from tensorflow.keras import utils  # внешние утилиты
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:

        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=190)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# x_train - изображения, y_train - правильный ответ в виде класса
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 60к изображений, каждое 28*28 пикселей (784 пикс)
x_train = x_train.reshape(60000, 784)

# x_train /= 255  # нормализация данных

# преобразуем метки классов на изображениях в категории, 10 классов
y_train = utils.to_categorical(y_train, 10)

model = Sequential()  # создаем последовательную модель

# добавляем слой нейронов состоящий из 800 нейронов и 784 входа в каждом
model.add(Dense(800, input_dim=784, activation="relu"))
# добавляем выходной слой состоящий из 10 нейронов (классы)
model.add(Dense(10, activation="softmax"))

# компилируем нейронную сеть используя оптимизацию градиентного спуска
model.compile(loss="categorical_crossentropy", optimizer="SGD")

print(model.summary())  # печатаем информацию о нейронной сети

# тренируем нейронную сеть , 100 эпох, с выводом информации о ходе тренировки
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

model.save('my_model.h5')

result = model.predict(x_train[0])  # используем модель для предсказания

print(str(result))

#model = keras.models.load_model('my_model.h5')
