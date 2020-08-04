import warnings
warnings.simplefilter('ignore')
import imp
import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils
from keras import regularizers, optimizers
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import pandas as pd
#from keras.layers.advanced_activations import LeakyReLU
import pylab
from keras_preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True

#Set GPU1 to be used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

def remove_ext(fn):
    return os.path.splitext(fn)[0]

traindf=pd.read_csv('./urban/UrbanSound8Kext.csv', sep=';',dtype=str)
traindf["slice_file_name"]=traindf["slice_file_name"].apply(append_ext)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.35)

train_datagen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.3, fill_mode="constant", cval=0)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./urban/DS/train/",
    x_col="slice_file_name",
    y_col="class",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(220,220))

valid_generator=validation_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./urban/DS/val/",
    x_col="slice_file_name",
    y_col="class",
    batch_size=64,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    target_size=(220,220))
'''
train_generator=train_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./urban/trainExt",
    x_col="slice_file_name",
    y_col="class",
    subset="training",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(220,220))

valid_generator=validation_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./urban/trainExt/",
    x_col="slice_file_name",
    y_col="class",
    subset="validation",
    batch_size=64,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(220,220))
'''

cnt = Counter(train_generator.classes)
print(cnt)
cnt = Counter(valid_generator.classes)
print(cnt)

input_shape=(220,220,3)

# Create model
'''
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu", name='dense_512'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(512, activation="relu", name='dense_512_2'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax", name='dense_out'),
])
'''
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

#optimizer = tensorflow.keras.optimizers.Adam(lr=1e-4)
#optimizer = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)
optimizer = keras.optimizers.Nadam(learning_rate=0.0005)
model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=8, min_lr=0.001, verbose=1)
mcp_save = ModelCheckpoint("./models/modelB2_weights_80_nadam.h5", save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='min', restore_best_weights=True)

class_weight = {0: 1.,
                1: 2.,
                2: 1.,
                3:1.,
                4:1.,
                5:1.,
                6:3.,
                7:1.,
                8:1.,
                9:1.}

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=80,
                    class_weight=class_weight,
                    callbacks=[reduce_lr, mcp_save, early_stop]
)

pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig('./models/rep/report_modelB2_80_nadam_acc.png')

pyplot.clf()
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='test_loss')
pyplot.legend()
pyplot.savefig('./models/rep/report_modelB2_80_nadam_loss.png')
pyplot.clf()

scores = model.evaluate(valid_generator, steps=STEP_SIZE_VALID)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
model.save('./models/modelB2_80_nadam.h5')

print("Model saved")

'''
optimizer = tensorflow.keras.optimizers.RMSprop(lr=0.0005, decay=1e-6)
model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=8, min_lr=0.001, verbose=1)
mcp_save = ModelCheckpoint("./models/modelB_weights_100_rms.h5", save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=100,
                    class_weight=class_weight,
                    callbacks=[reduce_lr, mcp_save]
)

pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig('./models/rep/report_modelB_100_rms_acc.png')

pyplot.clf()
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='test_loss')
pyplot.legend()
pyplot.savefig('./models/rep/report_model_100B_rms_loss.png')
pyplot.clf()

scores = model.evaluate(valid_generator, steps=STEP_SIZE_VALID)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
model.save('./models/model_100B_rms.h5')
print("Model saved")
'''
