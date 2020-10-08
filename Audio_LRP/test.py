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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, 10-0.5)
    plt.ylim(10-0.5, -0.5)
    return ax

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

testdf=pd.read_csv('../UrbanSound8K.csv', sep=',',dtype=str)
testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

cats= ["engine_idling","siren", "car_horn", "drilling", "gun_shot", "street_music", "jackhammer", "air_conditioner", "children_playing", "dog_bark"]
size = [ 1000, 929, 429, 1000, 374, 1000, 1000, 1000, 1000, 1000]

sizeCounter = 0
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_dataframe(
dataframe=testdf,
directory="../../mel/DS/test/",
x_col="slice_file_name",
y_col="class",
batch_size=1,
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

# Plot normalized confusion matrix
plot_confusion_matrix(test_generator.classes, y_pred, cats, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('../../mel/reports/normalized_confusion_matrix.png')

print('Classification Report')
target_names = cats
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

f =  open("../../mel/reports/classification_report.txt", "w")
f.write(classification_report(test_generator.classes, y_pred, target_names=target_names))
f.close()
