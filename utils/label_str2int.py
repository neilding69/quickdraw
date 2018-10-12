import os.path as path
import pickle

data_path = '/raid5/liuchang/quick_draw_output'

with open(path.join(data_path, 'data_Y'), 'rb') as f:
    Y = pickle.load(f)

Yu = set(Y)
dict = {}
for index, label in enumerate(Yu):
    dict[label] = index

Yc = []
for i in Y:
    Yc.append(dict[i])

with open(path.join(data_path, 'data_Y_int'), 'wb') as f:
    pickle.dump(Yc, f)
with open(path.join(data_path, 'dict_label_str2int'), 'wb') as f:
    pickle.dump(dict, f)
