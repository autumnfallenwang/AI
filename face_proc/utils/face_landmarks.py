import os
import os.path as osp
import sys
import glob
import cv2
import numpy as np
import face_alignment

sys.path.append('../')
from text_proc.utils.io import *


def bgr2rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_ldmk(img_rgb, device):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      flip_input=False,
                                      device=device)

    preds = fa.get_landmarks(img_rgb)

    if preds is not None:

    # 选取单个脸部ldmk，依据未知
        ldmk = np.asarray(preds)
        ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]

        return ldmk
    else:
        return None
    """
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
    """


def crop_with_ldmk(img, ldmk, crop_size, scale):

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


def face_coordinate(img, scale, device):
    h, w, _ = img.shape

    ldmk = get_ldmk(bgr2rgb(img), device)

    if ldmk is not None:

        ct_x, std_x = ldmk[:,0].mean(), ldmk[:,0].std()
        ct_y, std_y = ldmk[:,1].mean(), ldmk[:,1].std()

        std_x, std_y = scale * std_x, scale * std_y

        left = int(ct_x - std_x)
        if left < 0:
            left = 0
        top = int(ct_y - std_y)
        if top < 0:
            top = 0
        right = int(ct_x + std_x)
        if right >= w:
            right = w - 1
        bottom = int(ct_y + std_y)
        if bottom >= h:
            bottom = h - 1

        face_crop = img[top:bottom, left:right, :]
        return [left, top, right, bottom], face_crop
    else:
        return None, None


def save_crop_face(image_root, image_list, save_root, crop_size, scale, device):
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i, image_path in enumerate(image_list):
        _, image_name = osp.split(image_path)
        image_abspath = osp.join(image_root, image_path)

        img = cv2.imread(image_abspath)

        ldmk = get_ldmk(bgr2rgb(img), device)

        if ldmk is not None:
            crop_img = crop_with_ldmk(img, ldmk, crop_size, scale)

            save_path = osp.join(save_root, image_name)
            cv2.imwrite(save_path, crop_img)

            if i % 10 == 0 and i > 0:
                print('| Cropped [%d/%d] Images' % (i, len(image_list)))


def save_crop_face_from_image_root(image_root, label_file, image_type, save_root, crop_size, scale, device):
    print('INFO')
    print('-' * 80)
    print('| Image Root: %s' % image_root)
    print('| Save Root: %s' % save_root)
    print('| Crop Size: %s' % crop_size)
    print('| Scale: %s' % scale)
    print('| Device: %s' % device)
    print()
    print('Cropping...')
    print('-' * 80)
    if label_file is None:
        image_list = get_legal_root_list(image_root, image_type)
    else:
        image_list, _ = get_image_label_list(label_file, False)
        image_list = keep_legal_exts(image_list, image_type)
    save_crop_face(image_root, image_list, save_root, crop_size, scale, device)


def save_crop_face_from_video(video_path, save_root,
                              frame_interval=1,
                              crop_size=256,
                              scale=2.0,
                              device='cpu'):

    os.makedirs(save_root, exist_ok=True)
    _, video_file = osp.split(video_path)
    video_name, _ = osp.splitext(video_file)
    save_frame_root = osp.join(save_root, video_name)
    os.makedirs(save_frame_root, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("Load Video Fail")

    while success:
        success, frame = cap.read()
        if frame_index % frame_interval == 0 and success:
            ldmk = get_ldmk(bgr2rgb(frame), device)

            if ldmk is not None:
                crop_frame = crop_with_ldmk(frame, ldmk, crop_size, scale)

                save_path = osp.join(save_frame_root, str(frame_index)+'.jpg')
                cv2.imwrite(save_path, crop_frame)

                frame_count += 1
                if frame_count % 100 == 0:
                    print('| Cropped [%d] Frames' % frame_count)

        frame_index += 1

    cap.release()


def save_crop_face_from_video_folder(video_root, save_root,
                                     video_type=['.mp4', '.avi', '.mov'],
                                     frame_interval=1,
                                     crop_size=256,
                                     scale=2.0,
                                     device='cpu'):

    videos = get_root_list(video_root)
    videos = keep_legal_exts(videos, video_type)

    for i, video in enumerate(videos):
        print("| Crop [%d/%d] Video: %s" % (i+1, len(videos), video))

        save_crop_face_from_video(video_path=osp.join(video_root, video),
                                  save_root=save_root,
                                  frame_interval=frame_interval,
                                  crop_size=crop_size,
                                  scale=scale,
                                  device=device)
