import warnings
warnings.simplefilter('ignore')
import imp
import gc
import matplotlib.pyplot as plot
import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils
from keras import regularizers, optimizers
import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
import pandas as pd
import pylab
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from PIL import ImageFile
from sklearn.metrics import classification_report, confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

def remove_ext(fn):
    return os.path.splitext(fn)[0]
print("Loading model: ")
# Load model, including its weights and the optimizer
model = keras.models.load_model('../../mel/model/model_80_nadam.h5')
#optimizer = keras.optimizers.Nadam(learning_rate=0.0005)
#model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
#model.load_weights('./models/modelkB_weights_80_nadam.h5')
# Show the model architecture
model.summary()

testdf=pd.read_csv('../UrbanSound8Kext.csv', sep=';',dtype=str)
testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

cats= ["engine_idling","siren", "car_horn", "drilling", "gun_shot", "street_music", "jackhammer", "air_conditioner", "children_playing", "dog_bark"]
size = [ 1000, 929, 429, 1000, 374, 1000, 1000, 1000, 1000, 1000]

sizeCounter = 0
test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="../../mel/DS/test/",
x_col="slice_file_name",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode="categorical",
target_size=(220,220))

label_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="../../mel/DS/label_gen",
x_col="slice_file_name",
y_col="class",
batch_size=2,
seed=64,
shuffle=False,
class_mode="categorical",
target_size=(220,220))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()

Y_pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST,verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = cats
print(classification_report(test_generator.classes, y_pred, target_names=target_names))