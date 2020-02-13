import os, sys
import os.path as osp
import cv2

sys.path.append('../')
from text_proc.utils.io import *


def get_face_eye_list(face_label_file, split=', '):
    frame_id = []
    face = []
    eye = []
    with open(face_label_file, 'r') as read_file:
        while True:
            lines = read_file.readline()
            if not lines:
                break
            st = lines.rstrip('\n').split(split)
            frame_id.append(int(st[0]))
            if len(st) == 1:
                face.append([])
                eye.append([])
            else:
                face.append(list(map(int, [st[1], st[2], st[3], st[4]])))
                eye.append(list(map(int, [st[5], st[6], st[7], st[8]])))
    return frame_id, face, eye


def crop_with_face_coor(img, face_coor):
    h, w, _ = img.shape
    if face_coor != []:

        left = face_coor[0]
        if left < 0:
            left = 0
        top = face_coor[1]
        if top < 0:
            top = 0
        right = face_coor[2]
        if right >= w:
            right = w - 1
        bottom = face_coor[3]
        if bottom >= h:
            bottom = h - 1

        face_crop = img[top:bottom, left:right, :]
        return face_crop
    else:
        return None


def save_crop_face_from_video(video_path, face_label_file, save_root, frame_interval=1):

    os.makedirs(save_root, exist_ok=True)
    _, video_file = osp.split(video_path)
    video_name, _ = osp.splitext(video_file)
    save_frame_root = osp.join(save_root, video_name)
    os.makedirs(save_frame_root, exist_ok=True)

    frame_id, face, eye = get_face_eye_list(face_label_file)
    frame_num = len(frame_id)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("Load Video Fail")

    while success and (frame_index < frame_num):
        success, frame = cap.read()
        if frame_index % frame_interval == 0 and success:

            crop_frame = crop_with_face_coor(frame, face[frame_index])

            if crop_frame is not None:
                save_path = osp.join(save_frame_root, '%04d' % frame_index+'.jpg')
                cv2.imwrite(save_path, crop_frame)

                frame_count += 1
                if frame_count % 50 == 0:
                    print('| Cropped [%d] Frames' % frame_count)

        frame_index += 1

    cap.release()


def save_crop_face_from_video_folder(video_root, save_root,
                                     video_type=['.mp4', '.avi', '.mov'],
                                     frame_interval=1):

    videos = get_root_list(video_root)
    videos = keep_legal_exts(videos, video_type)

    for i, video in enumerate(videos):
        print("| Crop [%d/%d] Video: %s" % (i+1, len(videos), video))

        video_name, _ = osp.splitext(video)

        save_crop_face_from_video(video_path=osp.join(video_root, video),
                                  face_label_file = osp.join(video_root, video_name+'.face'),
                                  save_root=save_root,
                                  frame_interval=frame_interval)