from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from common.utils import get_train_valid_test_split

label_file = '/workspace/wangqiushi/datasets/d8_region_mix/xr_normal/xr_normal_old/label_xr_normal.txt'
shuffle = True
split_rate = [0.9, 0.1, 0.0]
select_rate = 1

# 1.00 0.71 0.50 0.35 0.25 0.18 0.13 0.09 0.06 0.04 0.03 0.02 0.015 0.01

# 1e0 1e-1 1e-2 1e-3 1e-4

save_dir = '/workspace/wangqiushi/datasets/d8_region_mix/xr_normal/xr_normal_old/t90/'
save_path = [save_dir+'train.txt',
             save_dir+'valid.txt',
             save_dir+'test.txt']

for f in save_path:
    if os.path.exists(f):
        os.remove(f)

get_train_valid_test_split(label_file=label_file,
                           shuffle=shuffle,
                           split_rate=split_rate,
                           select_rate=select_rate,
                           save_path=save_path)
