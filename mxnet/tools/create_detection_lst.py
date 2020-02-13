from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from common.utils import create_detection_lst_from_label_file


image_root = '/data02/wangqiushi/datasets/DR/Images/'
label_root = '/data02/wangqiushi/datasets/DR/mxnet_rec/t90/'
label_file = [label_root+'train.txt',
              label_root+'valid.txt']

map_file = label_root+'../DR_map.txt'
save_root = label_root

for f in label_file:
    if os.path.exists(f.replace('txt', 'lst')):
        os.remove(f)

for f in label_file:
    create_detection_lst_from_label_file(image_root=image_root,
                                         label_file=f,
                                         map_file=map_file,
                                         save_root=save_root)
