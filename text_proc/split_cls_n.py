import os.path as osp
import math
from utils.label_class import ClsLabel


label_path = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/811_mix/train.csv'
split_rate = 0.1
random_seed = 233
save_root = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/811_mix/'
balance_train = True
balance_method = 'oversample_times'


split_num = math.floor(1.0 / split_rate) - 1
split_list = []
whole = 1.0
for i in range(split_num):
    r  = split_rate / whole
    split_list.append(r)
    whole = whole-0.1

cl = ClsLabel(label_path)

train_cl_list = [cl]
train_cl = cl
for rate in split_list:
    train_cl, test_cl = train_cl.split(rate, random_seed)
    train_cl_list.append(train_cl)

if balance_train:
    for i, tcl in enumerate(train_cl_list):
        tcl = tcl.balance(method=balance_method, random_seed=random_seed)
        train_cl_list[i] = tcl

for i, tcl in enumerate(train_cl_list):
    print(tcl.count())
    n_str = str(len(train_cl_list)-i)
    tcl.to_csv(osp.join(save_root, 'train_'+n_str+'.csv'))
