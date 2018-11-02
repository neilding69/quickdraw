"""
Author: Liu Chang

This code processes the simplified dataset, downloaded from https://console.cloud.google.com/storage/quickdraw_dataset/full/simplified

The simplification process (which has been done by google) was
1.Align the drawing to the top-left corner, to have minimum values of 0.
2.Uniformly scale the drawing, to have a maximum value of 255.
3.Resample all strokes with a 1 pixel spacing.
4.Simplify all strokes using the Ramer–Douglas–Peucker algorithm with an epsilon value of 2.0.

This code will select the data from dataset according to the label and country code.

Kindly note that the data has been shuffled.

To use this code,
1.download the data
2.install ndjson
        pip install ndjson
3.change the value of dataset_path, class_name, country_code, output_path
4.run
5.look for data_X and data_Y in your output_path
"""

import glob
import os.path as path

import ndjson

dataset_path='/raid5/liuchang/quick_draw'
class_name = ['calendar', 'snowman', 'penguin', 'blackberry', 'teddy-bear']  # class_name=-1 means all class
country_code = -1  # country_code=-1 means all countries
output_path='/raid5/liuchang/quick_draw_output'


import numpy as np
def split_data(data):
    return [(i['recognized'],i['drawing'],i['word']) for i in data]

splitted=[]
i =0
min_len=np.inf
for file_path in glob.iglob(path.join(dataset_path,'*.ndjson')):
    i += 1
    label=file_path.split('/')[-1].split('.')[0]
    if class_name!=-1 and label not in class_name:
        continue
    print('readding data from {}'.format(file_path))
    with open(file_path) as f:
        data = ndjson.load(f)
    print('Now found {} avaliable data.'.format(len(data)))
    splitted.append(split_data(data))
    min_len=min(min_len,len(data))


import random
dataset=[]
for i,data in enumerate(splitted):
    dataset+=random.sample(data,min_len)
random.shuffle(dataset)
print(min_len,len(dataset))
Recognized=[i[0] for i in dataset]
X=[i[1] for i in dataset]
Y=[i[2] for i in dataset]

Yu = set(Y)
dict = {}
for index, label in enumerate(Yu):
    dict[label] = index

Yc = []
for i in Y:
    Yc.append(dict[i])

r=int(0.8*len(X))
train_Recognized=Recognized[:r]
train_X=X[:r]
train_Y=Yc[:r]
test_Recognized=Recognized[r:]
test_X=X[r:]
test_Y=Yc[r:]

import pickle
import os
with open(os.path.join(output_path,'1102_05b'+str(len(X))),'wb') as f:
    pickle.dump((train_Recognized,train_X,train_Y,test_Recognized,test_X,test_Y),f)