"""
Author: ZHANG Yu

The code is the functions will be used in lstm.ipynb and lstm_cnn.ipynb

"""
import math
import time
import pandas as pd
import pickle
import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.switch_backend('agg')
import torch


def _cut_stroke(x,ratio=1):
    if ratio==1:
        return x
    return [i[:int(len(i)*ratio)] for i in x]


def get_dataset(dataset_path,dataset_name,test_r1,test_r2,test_r3):
    (train_Recognized, train_X, train_Y, test_Recognized, test_X, test_Y)=pickle.load(open(os.path.join(dataset_path,dataset_name),'rb'))
    #train_X=_cut_stroke(train_X,train_r)
    #test_X0=_cut_stroke(test_X,test_r)
    test_X=test_X[:20000]
    test_X1=_cut_stroke(test_X,test_r1)
    test_X2=_cut_stroke(test_X,test_r2)
    test_X3=_cut_stroke(test_X,test_r3)
    return train_X[:100000],train_Y[:100000],test_X,test_Y[:20000],test_X1,test_X2,test_X3
 
    
    
def find_max_strok_point(data1,data2):      
    stroke_no1 = np.zeros(len(data1))    
    point_no1 = []
    for i in range(len(data1)): # number of pictures, ith picture            
        stroke_no1[i] = len(data1[i])
        for j in range(len(data1[i])): # number of strokes for each picture, jth stroke      
            point_no1.append(len(data1[i][j][0]))

    stroke_no_max1 = int(max(stroke_no1))
    point_no_max1 = int(max(point_no1))

    print ('max stroke number in train data =',stroke_no_max1,'\n max point number in train data =', point_no_max1)
    print ('training data number =', len(data1))
  
    stroke_no2 = np.zeros(len(data2))
    point_no2 = []

    for i in range(len(data2)): # number of pictures, ith picture   
        stroke_no2[i] = len(data2[i])
        for j in range(len(data2[i])): # number of strokes for each picture, jth stroke      
            point_no2.append(len(data2[i][j][0]))
 
    stroke_no_max2 = int(max(stroke_no2))
    point_no_max2 = int(max(point_no2))

    print ('max stroke number in test data =',stroke_no_max2,'\n max point number in test data =', point_no_max2)
    print ('test data number =', len(data2))
  
    max_stroke=max(stroke_no_max1,stroke_no_max2)
    max_point=max(point_no_max1,point_no_max2)
    return max_stroke,max_point
    
    
    
def convert_to_zeropad(data,max_stroke,max_point):

    Xdata = np.zeros((len(data),max_stroke,2,max_point))

    for i in range(len(data)):  
        for j in range(len(data[i])):        
            Xdata[i][j][0][:len(data[i][j][0])]=data[i][j][0][:]
            Xdata[i][j][1][:len(data[i][j][0])]=data[i][j][1][:]

    Xdata_tensor = torch.Tensor(Xdata/255)
    Xdata_tensor = Xdata_tensor.permute(0, 1, 3, 2)
    Xdata_tensor = Xdata_tensor.reshape(len(data), max_stroke, 2*max_point)
    return np.array(Xdata_tensor)


def label_onehot(label):

    onehot = []
    for i in range(len(label)):
        if label[i]==0:
            onehot.append([1,0,0,0,0])
        elif label[i]==1:
            onehot.append([0,1,0,0,0])
        elif label[i]==2:
            onehot.append([0,0,1,0,0])
        elif label[i]==3:
            onehot.append([0,0,0,1,0])
        elif label[i]==4:
            onehot.append([0,0,0,0,1])

    y=np.array(onehot)
    return y

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    #%matplotlib inline
    plt.subplots()
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.subplots()
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



