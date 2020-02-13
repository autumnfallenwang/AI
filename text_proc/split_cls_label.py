import os.path as osp
from utils.label_class import ClsLabel


label_path = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/train_labels.csv'
split_rate = [0.8, 0.2, 0.0]
random_seed = 233
save_root = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/820_shf/'
balance_train = True
balance_method = 'oversample_times'


cl = ClsLabel(label_path)
print('| Label Counts:')
print(cl.count())

train_cl, valid_cl, test_cl = cl.split3(split_rate=split_rate, random_seed=random_seed)

if balance_train:
    train_cl = train_cl.balance(method=balance_method, random_seed=random_seed)
    print('| Balance Train Label Counts:')
    print(train_cl.count())

train_cl.to_csv(osp.join(save_root, 'train.csv'))
valid_cl.to_csv(osp.join(save_root, 'valid.csv'))
test_cl.to_csv(osp.join(save_root, 'test.csv'))

