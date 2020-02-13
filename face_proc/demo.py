import os
import os.path as osp
import sys

sys.path.append('../')
from text_proc.utils import *

images, labels = get_image_label_list('../text_proc/labels.txt', False)

print(images)