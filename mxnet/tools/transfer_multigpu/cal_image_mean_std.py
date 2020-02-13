from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import cal_image_stat_from_label_file

label_file = '/workspace/wangqiushi/datasets/d8_region_mix/label_mix.txt'
image_dir = '/workspace/wangqiushi/datasets/d8_region_mix/raw_256_256'

cal_image_stat_from_label_file(label_file, image_dir)
