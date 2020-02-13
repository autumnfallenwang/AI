from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.utils import convert_model_to_onnx
from common.params import MULTI_GPU, GPU_LIST


model_path = '/workspace/wangqiushi/pytorch/models/saved_models/digestive8/t80/resnet50-dg8_nomean_pth040-20180829025439.pth'
class_num = 8
save_dir = './'
# model stucture saved in common.params

convert_model_to_onnx(model_path=model_path,
                      class_num=class_num,
                      multi_gpu=MULTI_GPU,
                      gpu_list=GPU_LIST,
                      save_dir=save_dir)