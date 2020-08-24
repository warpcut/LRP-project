import split_folders
import imp
import os
import pandas as pd
from shutil import copyfile

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
def split_dataset(fromPath, toPath):
    split_folders.ratio(fromPath, output=toPath, seed=1337, ratio=(.7, .2, .1))

def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

def refold_images(fromPath, toPath)
    testdf=pd.read_csv('./urban/UrbanSound8K.csv',dtype=str)
    testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

    for index, row in testdf.iterrows():
    #print(row['slice_file_name'], row['class'])
    copyfile(fromPath + str(row['slice_file_name']), toPath + str(row['class']) + "/" + str(row['slice_file_name']))
