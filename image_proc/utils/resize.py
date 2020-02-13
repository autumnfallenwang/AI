import os
import os.path as osp
import sys
import cv2

sys.path.append('../')
from text_proc.utils.io import get_legal_root_list


def resize_image_from_root(image_root, legal_exts, save_root, resize_size=(256, 256)):
    os.makedirs(save_root, exist_ok=True)
    image_list = get_legal_root_list(image_root, legal_exts)

    for i, image in enumerate(image_list):
        image_path = osp.join(image_root, image)
        image_save_path = osp.join(save_root, image)
        img = cv2.imread(image_path)
        img_resize = cv2.resize(img, resize_size)
        cv2.imwrite(image_save_path, img_resize)

        if i % 100 == 0 and i > 0:
            print('| Resize [%d/%d] Images' % (i, len(image_list)))

