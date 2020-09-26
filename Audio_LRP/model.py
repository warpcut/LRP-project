import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils
from keras import regularizers, optimizers
def get_keras_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), use_bias=False, input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, (3, 3), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(64, (3, 3), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(64, (3, 3), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(128, (3, 3), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Conv2D(128, (3, 3), use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(512, use_bias=False, name='dense_512'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(512, use_bias=False, name='dense_512_2'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, use_bias=False, name='dense_out'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("softmax")
    ])
    return model

def get_optimizer():
    optimizer = keras.optimizers.Nadam(learning_rate=0.0005)
    return optimizer
