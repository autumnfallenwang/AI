from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import mxnet as mx
from common.utils import plot_mask_with_gluon_model

image_root = '/Users/wangqiushi/pycode/test/ultrasonic_test/test_image/'
save_root = '/Users/wangqiushi/pycode/test/ultrasonic_test/test_res/'
model_name = 'fcn'
backbone = 'resnet101'
dataset = 'ultrasonic_voc'
label_file = '/Users/wangqiushi/pycode/test/ultrasonic_test/class_names.txt'
model_path = '/Users/wangqiushi/pycode/test/ultrasonic_test/checkpoint.params'

ctx = mx.cpu(0)

plot_mask_with_gluon_model(image_root=image_root, 
                            model_path=model_path, 
                            model_name=model_name, 
                            backbone=backbone, 
                            dataset=dataset, 
                            label_file=label_file, 
                            save_root=save_root, 
                            ctx=ctx)
