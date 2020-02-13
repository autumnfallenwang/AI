from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from common.utils import export_segmentation_model


model_name = 'fcn'
backbone = 'resnet101'
dataset = 'ultrasonic_voc'
model_path = '/Users/wangqiushi/pycode/SuperComputer/mxnet/projects/ultrasonic/seg/saved_models/fcn/res101_resume_lr0001/us_fcn_res101.params'
save_root = '/Users/wangqiushi/pycode/SuperComputer/mxnet/projects/ultrasonic/seg/export_models/'


export_segmentation_model(model_name=model_name, 
					      backbone=backbone, 
					      dataset=dataset, 
                          model_path=model_path,  
                          save_root=save_root)

