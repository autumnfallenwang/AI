import os
import shutil
from common.face_landmarks import save_crop_face_from_label_file


image_root =  '/home/wangqiushi/pycode/datasets/NUAA/images/'
label_file =  '/home/wangqiushi/pycode/datasets/NUAA/labels.txt'
save_root = '/home/wangqiushi/pycode/datasets/NUAA/faces_224/scale_2.0/'
crop_size =  224
scale = 2.0
device = 'cuda'

'''
if os.path.exists(save_root):
    shutil.rmtree(save_root)
'''

save_crop_face_from_label_file(image_root, label_file, save_root, crop_size, scale, device)
