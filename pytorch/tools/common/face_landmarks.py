import os
import os.path as osp
import sys
import glob
import cv2
import numpy as np
import face_alignment
from skimage import io
from .utils import get_image_label_list


def get_ldmk(image_path, device):
    img = io.imread(image_path)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      flip_input=False,
                                      device=device)

    preds = fa.get_landmarks(img)

    # 选取单个脸部ldmk，依据未知
    ldmk = np.asarray(preds)
    ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
    '''
    if 0:
        for pred in preds:
            img = cv2.imread(imgdir)
            print('ldmk num:', pred.shape[0])
            for i in range(pred.shape[0]):
                x,y = pred[i]
                print(x,y)
                cv2.circle(img,(x,y),1,(0,0,255),-1)
            cv2.imshow('-',img)
            cv2.waitKey()
    '''
    return ldmk


def crop_with_ldmk(image_path, ldmk, crop_size, scale):
    img = cv2.imread(image_path)

    ct_x, std_x = ldmk[:,0].mean(), ldmk[:,0].std()
    ct_y, std_y = ldmk[:,1].mean(), ldmk[:,1].std()

    std_x, std_y = scale * std_x, scale * std_y

    src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
    dst = np.float32([((crop_size - 1) / 2.0, (crop_size - 1) / 2.0),
                      ((crop_size - 1), (crop_size - 1)),
                      ((crop_size - 1), (crop_size - 1) / 2.0)])
    retval = cv2.getAffineTransform(src, dst)
    result = cv2.warpAffine(img, retval, (crop_size, crop_size),
                            flags = cv2.INTER_LINEAR,
                            borderMode = cv2.BORDER_CONSTANT)
    return result


def save_crop_face(image_root, label_file, save_root, crop_size, scale, device):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    image_list, _ = get_image_label_list(label_file, False)    

    for i, image_path in enumerate(image_list):
        _, image_name = osp.split(image_path)
        image_abspath = osp.join(image_root, image_path)

        ldmk = get_ldmk(image_abspath, device)
        crop_img = crop_with_ldmk(image_abspath, ldmk, crop_size, scale)

        save_path = osp.join(save_root, image_name)
        cv2.imwrite(save_path, crop_img)

        if i % 10 == 0 and i > 0:
            print('| Cropped [%d/%d] Images' % (i, len(image_list)))


def save_crop_face_from_label_file(image_root, label_file, save_root, crop_size, scale, device):
    print('INFO')
    print('-' * 80)
    print('| Image Root: %s' % image_root)
    print('| Label File: %s' % label_file)
    print('| Save Root: %s' % save_root)
    print('| Crop Size: %s' % crop_size)
    print('| Scale: %s' % scale)
    print('| Device: %s' % device)
    print()
    print('Cropping...')
    print('-' * 80)

    save_crop_face(image_root, label_file, save_root, crop_size, scale, device)