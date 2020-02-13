from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from common.utils import get_train_valid_test_split

label_file = '/data02/wangqiushi/datasets/ultrasonic/seg/jiejie_label.txt'
attribute = 'cls'
# attribute = 'det'
shuffle = True
split_rate = [0.8, 0.1, 0.1]
select_rate = 1

save_dir = '/data02/wangqiushi/datasets/ultrasonic/seg/voc_like_data/jiejie/ImageSets/t80/'
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
