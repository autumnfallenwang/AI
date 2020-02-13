import os
import os.path as osp
import shutil

from utils.resize import resize_image_from_root


image_root = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/test/'
legal_exts = ['.jpg', '.jpeg', '.png']
save_root = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/train_test_256/'
resize_size = (256, 256)


resize_image_from_root(image_root, legal_exts, save_root, resize_size)
