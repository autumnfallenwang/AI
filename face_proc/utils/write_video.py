import os, sys
import os.path as osp
import shutil
import copy
import cv2
# import mxnet as mx

from .face_landmarks import bgr2rgb, face_coordinate
from .test_mxnet_symbol_model import predict_image
from .face_labels import get_face_eye_list

sys.path.append('../')
from text_proc.utils.io import *


def draw_line(img, p1, p2, colour=(0,0,255), thickness=2):
    cv2.line(img, p1, p2, colour, thickness)
    return img


def draw_rectangle(img, p1, p2, colour=(0,0,255), thickness=2):
    cv2.rectangle(img, p1, p2, colour, thickness)
    return img


def draw_circle(img, center, radius, colour=(0,0,255), thickness=2):
    cv2.circle(img, center, radius, colour, thickness)
    return img


def draw_text(img, text,
              p1=(10, 30),
              text_size=1,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              colour=(0,0,255), 
              thickness=2):

    cv2.putText(img, text, p1, font, text_size, colour, thickness)
    return img


def write_to_video(video_path, model_path,
                   save_path,
                   scale=3.0,
                   device='cpu'):

    cap = cv2.VideoCapture(video_path)
    save_video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    save_video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), save_video_fps, save_video_size)

    cap_frame_index = 0

    if cap.isOpened():
        success = True
    else:
        success = False
        print("Load Video Fail")

    while success:
        success, frame = cap.read()

        if success:
            if cap_frame_index % 50 == 0:
                print('| frame: %d' % cap_frame_index)

            face_coor, face_crop = face_coordinate(frame, scale, device)

            if face_coor is not None:
                pred = predict_image(model_path, bgr2rgb(face_crop), device)

                if pred[0] == 0:
                    draw_rectangle(frame, (face_coor[0], face_coor[1]), (face_coor[2], face_coor[3]))
                elif pred[0] == 1:
                    draw_rectangle(frame, (face_coor[0], face_coor[1]), (face_coor[2], face_coor[3]), colour=(0,255,0))

                frame = draw_text(frame, 'C: '+str(pred[0]))
                frame = draw_text(frame, 'P: '+('%.4f' % pred[1]), (150,30))

            video.write(frame)
            cap_frame_index += 1

    cap.release()
    video.release()


def draw_face_eye(orig_video_path, face_label_file, save_draw_path, save_empty_path):

    frame_id, face, eye = get_face_eye_list(face_label_file)
    frame_num = len(frame_id)

    video_orig = cv2.VideoCapture(orig_video_path)
    save_video_fps = int(video_orig.get(cv2.CAP_PROP_FPS))
    save_video_size = (int(video_orig.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_orig.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video_draw = cv2.VideoWriter(save_draw_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), save_video_fps, save_video_size)
    video_empty = cv2.VideoWriter(save_empty_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), save_video_fps, save_video_size)
    
    frame_index = 0

    if video_orig.isOpened():
        success = True
    else:
        success = False
        print("Load Orig Video Fail")

    while success and (frame_index < frame_num):
        success, frame = video_orig.read()
        frame_empty = copy.copy(frame)

        if success:
            if frame_index % 50 == 0:
                print('| Frame: %d' % frame_index)

        face_coor = face[frame_index]
        eye_coor = eye[frame_index]

        if face_coor != []:
            draw_rectangle(frame, (face_coor[0], face_coor[1]), (face_coor[2], face_coor[3]))
            draw_circle(frame, (eye_coor[0], eye_coor[1]), radius=2, thickness=-1)
            draw_circle(frame, (eye_coor[2], eye_coor[3]), radius=2, thickness=-1)


        video_draw.write(frame)
        video_empty.write(frame_empty)
        frame_index += 1

    video_orig.release()
    video_draw.release()
    video_empty.release()


def draw_face_eye_videos(video_face_root, video_type, output_empty_root):
    os.makedirs(output_empty_root, exist_ok=True)
    video_list = get_legal_root_list(video_face_root, video_type)
    for i, video in enumerate(video_list):
        print("| Draw [%d/%d] Video: %s" % (i+1, len(video_list), video))

        video_name, _ = osp.splitext(video)
        orig_video_path = osp.join(video_face_root, video)
        face_label_file = osp.join(video_face_root, video_name+'.face')
        save_draw_path = osp.join(output_empty_root, video_name+'_face.avi')
        save_empty_path = osp.join(output_empty_root, video_name+'.avi')

        draw_face_eye(orig_video_path, face_label_file, save_draw_path, save_empty_path)
        shutil.copy(face_label_file, output_empty_root)