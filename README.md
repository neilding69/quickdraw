# Quickdraw
GroupID: Group 3

## Description: 
Quick, Draw! is an online game developed by Google that challenges players to draw a picture of an object or idea and then uses a neural network artificial intelligence to guess what the drawings represent.

## Motivation:
To build models that make prediction out of Complete as well as Incomplete sketches and compare the results.

## Author
1. Liu Chang
2. Zhang Yu
3. Qiu Yang
4. Ding Shuya

## To Run the codes
1. Get raw data: 1102_05b633244.dms
https://drive.google.com/drive/folders/1ovQX5gqle7JOL7RcO6f_05n7DW2FrOUk?usp=sharing

2. Run DatasetGeneration.ipynb in the utils folder, to generate data files. use lines in read_data.py in your code to load these files.

3. Pick a model that you want to run, change the data path and you should be ready to go.

All codes and other data are also available at git clone https://github.com/neilding69/quickdraw

## Folder Structure
1. Person_Modelname.ipynb：model reproduce 
2. Data Analysis: data analysis of quick draw
3. Utils folder: data generation and preprocessing file 
4. Toy_dataset folder: some toy dataset (used in qiuyang_lstm_cnn.ipynb)

## Model Structure 
1. CNN: dingshuya_cnn.ipynb
2. LSTM: zhangyu_lstm.ipynb
3. CNN-GRU:liuchang_CNN_GRU.ipynb
4. LSTM-CNN: dingshuya_lstm_cnn.ipynb
5. LSTM-CNN: zhangyu_lstm_cnn.ipynb
6. LSTM-CNN: qiuyang_lstm_cnn.ipynb
7. CNN-LSTM-CNN: qiuyang_cnn_lstm_cnn.ipynb

## Results:
![alt text](https://github.com/neilding69/quickdraw/blob/master/chart.png)
