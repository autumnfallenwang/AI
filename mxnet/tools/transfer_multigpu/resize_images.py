from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import resize_image_from_label_file

label_file = '/workspace/wangqiushi/datasets/d8_region_mix/label_mix.txt'
image_dir = '/'
save_dir = '/workspace/wangqiushi/datasets/d8_region_mix/raw_256_256/'
resize_size = (256, 256)

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)

resize_image_from_label_file(label_file=label_file,
                             image_dir=image_dir,
                             save_dir=save_dir,
                             resize_size=resize_size)