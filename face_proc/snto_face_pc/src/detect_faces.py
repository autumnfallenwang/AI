import os, sys, argparse
import os.path as osp
import cv2
import numpy as np

import utils
from sntoFace import SntoFace 


save_from = ['image_folder', 'single_video', 'video_folder']
save_type = save_from[2]

"""
# image_folder
image_root = '/u2t/wangqiushi/datasets/NUAA/raw/ClientRaw/0001/'
#image_root = '/u2t/wangqiushi/datasets/catdog/cat/'
save_file = './images.face'
image_type=['.jpg']
split=', '
"""
"""
# single_video
video_path = '/u2t/wangqiushi/datasets/MSU-MFSD/MSU-MFSD/scene01/attack/attack_client055_laptop_SD_printed_photo_scene01.mov'
save_root = './'
split=', '
frame_interval=1
"""
# video_folder

video_root = '/u2t/wangqiushi/datasets/SNTO_test/snto_test_20191106/split/split_cellphone/' 
save_root = '/u2t/wangqiushi/datasets/SNTO_test/snto_test_20191106/split/split_cellphone/'
video_type=['.mp4', '.avi', '.mov']
split=', '
frame_interval=1



parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--re-model', default='', help='path to load model.')
#parser.add_argument('--mt-model', default='/u2t/share/insightface/model/FaceDetection/mtcnn-model/model_mxnet', help='path to load model.')
parser.add_argument('--mt-model', default='./model_mxnet', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()

snto_model = SntoFace(args)


def get_root_list(root):
    files = []
    for root, dirs, file in os.walk(root):  
    #    print(root) #当前目录路径 
    #    print(dirs) #当前路径下所有子目录
    #    print(file)
        files.append(file)
    return files[0]


def keep_legal_exts(file_list, legal_exts):
    keep_list = []
    for file in file_list:
        if osp.splitext(file)[1].lower() in legal_exts:
            keep_list.append(file)
    if len(keep_list) == 0:
        print('No Legal File in List')
    return keep_list


def detect_one_face_image(img, select='max_area'):
    bbox, points = snto_model.mtcnn_detect(img)
    """
    bbox = [xmin, ymin, xmax, ymax, p]
    points = [l_eye_x, r_eye_x, nose_x, l_mouth_x, r_mouth_x, l_eye_y, ...]
    """
    if len(bbox) == 0:
        return [], []
    else:
        if select == 'max_area':
            area = (bbox[:,2]-bbox[:,0])*(bbox[:,2]-bbox[:,0])
            face_id = np.argsort(area)[-1]
        return bbox[face_id], points[face_id]


def detect_face_eye(img):
    """
    face_eye = [xmin, ymin, xmax, ymax, l_eye_x, l_eye_y, r_eye_x, r_eye_y]
    """
    face_eye = np.array([])
    bbox, points = detect_one_face_image(img)
    if len(bbox) != 0:
        bbox = np.rint(bbox).astype(np.int)
        face_eye = np.array([bbox[0], bbox[1], bbox[2], bbox[3], points[0], points[5], points[1], points[6]])
    return face_eye


def save_faces_from_image_folder(image_root,
                                 save_file = './images.face',
                                 image_type=['.jpg'],
                                 split=', '):

    images = get_root_list(image_root)
    images = keep_legal_exts(images, image_type)

    faces = []
    for i, image in enumerate(images):
        img = cv2.imread(osp.join(image_root, image))
        face_eye = detect_face_eye(img)
        if len(face_eye) == 0:
            faces.append(image)
        else:
            face_eye = list(map(str, face_eye))
            face_eye.insert(0, image)
            faces.append(split.join(face_eye))

        if i % 100 == 0 and i > 0:
            print('| Detected [%d/%d] Images' % (i, len(images)))

    if len(faces) != 0:
        with open(save_file, 'w+') as f:
            for face in faces:
                f.write('%s\n' % face)


def save_faces_from_video(video_path, save_root, split=', ', frame_interval=1):

    os.makedirs(save_root, exist_ok=True)
    _, video_file = osp.split(video_path)
    video_name, _ = osp.splitext(video_file)
    save_file = osp.join(save_root, video_name+'.face')

    faces = []

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
            face_eye = detect_face_eye(frame)

            if len(face_eye) == 0:
                faces.append(str(frame_count))
            else:
                face_eye = list(map(str, face_eye))
                face_eye.insert(0, str(frame_count))
                faces.append(split.join(face_eye))

            frame_count += 1
            if frame_count % 100 == 0:
                print('| Detected [%d] Frames' % frame_count)

        frame_index += 1

    cap.release()

    if len(faces) != 0:
        with open(save_file, 'w+') as f:
            for face in faces:
                f.write('%s\n' % face)


def save_faces_from_video_folder(video_root, save_root,
                                 video_type=['.mp4', '.avi', '.mov'],
                                 split=', ',
                                 frame_interval=1):

    videos = get_root_list(video_root)
    videos = keep_legal_exts(videos, video_type)

    for i, video in enumerate(videos):
        print("| Detect [%d/%d] Video: %s" % (i+1, len(videos), video))

        save_faces_from_video(video_path=osp.join(video_root, video),
                              save_root=save_root,
                              split=split,
                              frame_interval=frame_interval)



if save_type == 'image_folder':
    save_faces_from_image_folder(image_root=image_root,
                                 save_file=save_file,
                                 image_type=image_type,
                                 split=split)

elif save_type == 'single_video':
    save_faces_from_video(video_path=video_path,
                          save_root=save_root,
                          split=split,
                          frame_interval=frame_interval)

elif save_type == 'video_folder':
    save_faces_from_video_folder(video_root=video_root,
                                 save_root=save_root,
                                 video_type=video_type,
                                 split=split,
                                 frame_interval=frame_interval)

"""
# CASIA-FASD
video_type=['.mp4', '.avi', '.mov']
split=', '
frame_interval=1

root = '/u2t/wangqiushi/datasets/CASIA-FASD/CASIA-FASD/'
v_sets = {'train_release':20, 'test_release':30}

for v_set in list(v_sets.keys()):
    print('| Set: %s' % v_set)
    for v in range(v_sets[v_set]):
        print('| Sub: %s' % str(v+1))
        video_root = osp.join(root, v_set, str(v+1))
        save_root = video_root

        save_faces_from_video_folder(video_root=video_root,
                                     save_root=save_root,
                                     video_type=video_type,
                                     split=split,
                                     frame_interval=frame_interval)
"""
