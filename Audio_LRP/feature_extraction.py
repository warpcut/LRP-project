from glob import glob
import numpy as np
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path

def create_mel_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = './mel/DS/imgs/' + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_cqt_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(clip, sr=sample_rate)), ref=np.max)
    librosa.display.specshow(CQT, y_axis='cqt_hz')
    filename  = './cqt/DS/imgs/' + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,CQT

def extract_spectrograms(fromPath):
    Data_dir=np.array(glob(fromPath))

    i=0
    for file in Data_dir[i:i+2000]:
        #Define the filename as is, "name" refers to the PNG, and is split off into the number itself.
        filename,name = file,file.split('/')[-1].split('.')[0]
        create_mel_spectrogram(filename,name)
        create_cqt_spectrogram(filename,name)
    gc.collect()

    i=2000
    for file in Data_dir[i:i+2000]:
        filename,name = file,file.split('/')[-1].split('.')[0]
        create_mel_spectrogram(filename,name)
        create_cqt_spectrogram(filename,name)
    gc.collect()

    i=4000
    for file in Data_dir[i:]:
        filename,name = file,file.split('/')[-1].split('.')[0]
        create_mel_spectrogram(filename,name)
        create_cqt_spectrogram(filename,name)
    gc.collect()
