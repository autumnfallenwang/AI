from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import export_detection_model


model_name = 'yolo3_darknet53_voc'
model_path = '/mnt/datasets02/wangqiushi/mxnet/projects/DR/saved_models/yolo3_darknet53_voc_best.params'
model_classes = ('30', '40', '50')
save_root = '/mnt/datasets02/wangqiushi/mxnet/projects/DR/export_models/'


export_detection_model(model_name=model_name, 
                       model_path=model_path, 
                       model_classes=model_classes, 
                       save_root=save_root)


