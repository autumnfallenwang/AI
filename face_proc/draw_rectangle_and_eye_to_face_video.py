import os, sys
import os.path as osp

from utils.write_video import draw_face_eye_videos


video_type = ['.avi']

"""
video_face_root = '/home/wangqiushi/Documents/face_spoofing_datasets/SNTO_test_20191106/split/paper2/'

output_empty_root = '/home/wangqiushi/Documents/face_spoofing_datasets/SNTO_test_20191106/split/paper2_proc/'

draw_face_eye_videos(video_face_root, video_type, output_empty_root)
"""


# CASIA-FASD
root = '/home/wangqiushi/pycode/face_spoofing/datasets/CASIA-FASD/CASIA-FASD/'
v_sets = {'train_release':20, 'test_release':30}
save_root = '/home/wangqiushi/pycode/face_spoofing/datasets/CASIA-FASD/draw_face/'

for v_set in list(v_sets.keys()):
    print('| Set: %s' % v_set)
    for v in range(v_sets[v_set]):
        print('| Sub: %s' % str(v+1))
        video_root = osp.join(root, v_set, str(v+1))
        save_empty_root = osp.join(save_root, v_set, str(v+1))

        draw_face_eye_videos(video_root, video_type, save_empty_root)
