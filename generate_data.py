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
class_name = ['calender', 'snowman', 'penguin', 'blackberry', 'teddy-bear']  # class_name=-1 means all class
country_code = ['JP', 'CN', 'DE']  # country_code=-1 means all countries
output_path='/raid5/liuchang/quick_draw_output'


def split_data(country_code,data):
    splitted=[]
    if isinstance(country_code,str):
        country_code=country_code.split(' ')
    for i in data:
        if country_code!=-1 and i['countrycode']  not in country_code:
            continue
        # if i['recognized']==False:
        #     continue
        splitted.append((i['drawing'],i['word']))
    return splitted

splitted=[]
i =0
for file_path in glob.iglob(path.join(dataset_path,'*.ndjson')):
    i += 1
    label=file_path.split('/')[-1].split('.')[0]
    if class_name!=-1 and label not in class_name:
        continue
    print('readding data from {}'.format(file_path))
    with open(file_path) as f:
        data = ndjson.load(f)
    # for i in data[:4]:
    #     print(i)
    splitted+=split_data(country_code,data)
    print('Now found {} avaliable data.'.format(len(splitted)))



import random
random.shuffle(splitted)

X=[i[0] for i in splitted]
Y=[i[1] for i in splitted]

import pickle
with open(path.join(output_path,'data_X'),'wb') as f:
    pickle.dump(X,f)

with open(path.join(output_path,'data_Y'),'wb') as f:
    pickle.dump(Y,f)