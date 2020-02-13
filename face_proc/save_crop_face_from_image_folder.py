import os
import shutil
from utils.face_landmarks import save_crop_face_from_image_root


image_root =  '/u2t/wangqiushi/datasets/SNTO_test/snto_test_20191101/'
# label_file =  './test_image_labels.txt'
label_file = None
image_type = ['.jpg']
save_root = '/u2t/wangqiushi/datasets/SNTO_test/snto_test_face_20191101/'
crop_size =  224
scale = 3.0
device = 'cuda'

'''
if os.path.exists(save_root):
    shutil.rmtree(save_root)
'''

save_crop_face_from_image_root(image_root, label_file, image_type, save_root, crop_size, scale, device)
