from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import mxnet as mx
from common.utils import plot_bbox_with_symbol_model

image_root = '/home/wangqiushi/Medical_Datasets/digestive8/test_bbox/test1/'
model_path = '/home/wangqiushi/pycode/SuperComputer/mxnet/projects/DR/moxing/dr/ssd_512_resnet50_v1_voc_best'
save_root = '/home/wangqiushi/Medical_Datasets/digestive8/test_bbox/test2/'
threshold = 0.2
classes = ('ml', 'xr', 'xsw')
ctx = mx.gpu(0)

plot_bbox_with_symbol_model(image_root=image_root, 
                            model_path=model_path, 
                            save_root=save_root, 
                            threshold=threshold, 
                            classes=classes, 
                            ctx=ctx)