import os
from utils.split import get_train_valid_test_split

label_file = '/home/wangqiushi/pycode/datasets/kaggle/labels/labels.txt'
attribute = 'cls'
# attribute = 'det'
shuffle = True
split_rate = [0.8, 0.1, 0.1]
select_rate = 1

save_dir = '/home/wangqiushi/pycode/datasets/kaggle/labels/'
save_path = [save_dir+'train.txt',
             save_dir+'valid.txt',
             save_dir+'test.txt']

for f in save_path:
    if os.path.exists(f):
        os.remove(f)

get_train_valid_test_split(label_file=label_file,
                           attribute=attribute,
                           shuffle=shuffle,
                           split_rate=split_rate,
                           select_rate=select_rate,
                           save_path=save_path)
