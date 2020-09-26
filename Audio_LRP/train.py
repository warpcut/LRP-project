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
import pylab
from keras_preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot
from collections import Counter

from model import get_keras_model, get_optimizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

#Set GPU1 to be used
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

def remove_ext(fn):
    return os.path.splitext(fn)[0]
    
# Generatori

traindf=pd.read_csv('./UrbanSound8Kext.csv', sep=';',dtype=str)
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
    
cnt = Counter(train_generator.classes)
print(cnt)
cnt = Counter(valid_generator.classes)
print(cnt)

input_shape=(220,220,3)

# Load Model

model = get_keras_model(input_shape)
optimizer = get_optimizer()

# Def callbacks

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

# Train

history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=80,
                    class_weight=class_weight,
                    callbacks=[reduce_lr, mcp_save, early_stop]
)

# Save plots

pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.savefig('./models/rep/report_model_80_nadam_acc.png')

pyplot.clf()
pyplot.plot(history.history['loss'], label='train_loss')
pyplot.plot(history.history['val_loss'], label='test_loss')
pyplot.legend()
pyplot.savefig('./models/rep/report_model_80_nadam_loss.png')
pyplot.clf()

scores = model.evaluate(valid_generator, steps=STEP_SIZE_VALID)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
f = open("./models/rep/report_modelB2_80_nadam.txt", "w")
f.write("Scores on test set: loss=%s accuracy=%s" % tuple(scores))
f.close()

# Save trained model
model.save('./models/modelB2_80_nadam.h5')

print("Model saved")
