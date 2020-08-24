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

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

def remove_ext(fn):
    return os.path.splitext(fn)[0]
print("Loading model: ")
# Load model, including its weights and the optimizer
model = keras.models.load_model('./models/modelkB_80_nadam.h5')
#optimizer = keras.optimizers.Nadam(learning_rate=0.0005)
#model.compile(optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
#model.load_weights('./models/modelkB_weights_80_nadam.h5')
# Show the model architecture
model.summary()


