from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import create_image_folder_from_label_file

image_root = '/raid/data/kaggle/diabetic_retinopathy/train/'
label_file = '/raid/data/kaggle/diabetic_retinopathy/labels/labels.txt'
save_root = '/raid/data/kaggle/diabetic_retinopathy/train_folder/'

if os.path.exists(save_root):
    shutil.rmtree(save_root)

create_image_folder_from_label_file(image_root, label_file, save_root)
