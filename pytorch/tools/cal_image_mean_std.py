from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import cal_image_stat_from_label_file

label_file = '/workspace/wangqiushi/datasets/digestive8/train.txt'
image_dir = '/workspace/wangqiushi/datasets/digestive8/train/'

cal_image_stat_from_label_file(label_file, image_dir)
