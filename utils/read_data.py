"""
Author: Liu Chang

Example:
>>> from utils.read_data import get_dataset
>>> dataset_path='/raid5/liuchang/quick_draw_output/'
>>> train_X,train_Y,test_X,test_Y=get_dataset(dataset_path,test_r=0.5)

Please note that the data has been shuffled.

The dataset has 5 classes: 'calendar', 'snowman', 'penguin', 'blackberry', 'teddy-bear'. Each class have exactly 128153 data.

In get_dataset(), train_r=(0,1] indicates how much you want to use in training dataset. Similarly for testy_r=(0,1]


"""
import os
import pickle
def get_dataset(dataset_path='/raid5/liuchang/quick_draw_output/',dataset_name='1102_05b640765',train_r=1.0,test_r=1.0):
    (train_Recognized, train_X, train_Y, test_Recognized, test_X, test_Y)=pickle.load(open(os.path.join(dataset_path,dataset_name),'rb'))
    train_r=int(train_r*len(train_X))
    test_r=int(test_r*len(test_X))
    return train_X[:train_r],train_Y[:train_r],test_X[:test_r],test_Y[:test_r]