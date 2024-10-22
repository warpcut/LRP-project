import imp
import os
import pandas as pd
from shutil import copyfile

def append_ext(fn):
    result = os.path.splitext(fn)[0]
    return result +".png"

testdf=pd.read_csv('../UrbanSound8K.csv', sep=',',dtype=str)
testdf["slice_file_name"]=testdf["slice_file_name"].apply(append_ext)

for index, row in testdf.iterrows():
  #print(row['slice_file_name'], row['class'])
  copyfile("../../cq/cq-train/" + str(row['slice_file_name']), "../../cq/DS/folded/" + str(row['class']) + "/" + str(row['slice_file_name']))
