from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import export_model

model_name = 'resnet50_v1d'
model_path = '/mnt/datasets02/wangqiushi/mxnet/projects/digestive8/saved_models/ResNet50_v1d--20181120150731.params'
class_num = 8
save_root = '/mnt/datasets02/wangqiushi/mxnet/projects/digestive8/export_models/'

export_model(model_name=model_name, 
             model_path=model_path, 
             class_num=class_num, 
             save_root=save_root)