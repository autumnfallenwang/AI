import os
import os.path as osp


#mode = 'face_landmarks'
mode = 'face_labels'

if mode == 'face_landmarks':
    from utils.face_landmarks import save_crop_face_from_video_folder


    video_type = ['.mp4', '.avi', '.mov']
    frame_interval = 1
    crop_size = 256
    scale = 3.0
    device = 'cpu'


    video_root = './faces/'
    save_root = './test/'
    save_crop_face_from_video_folder(video_root=video_root,
                                     save_root=save_root,
                                     video_type=video_type,
                                     frame_interval=frame_interval,
                                     crop_size=crop_size,
                                     scale=scale,
                                     device=device)

    """
    # CASIA-FASD
    root = '/u2t/wangqiushi/datasets/CASIA-FASD/CASIA-FASD/'
    v_sets = {'train_release':20, 'test_release':30}
    save_crop_root = '/u2t/wangqiushi/datasets/CASIA-FASD/shot_images/'

    for v_set in list(v_sets.keys()):
        print('| Set: %s' % v_set)
        for v in range(v_sets[v_set]):
            print('| Sub: %s' % str(v+1))
            video_root = osp.join(root, v_set, str(v+1))
            save_root = osp.join(save_crop_root, v_set, str(v+1))

            save_crop_face_from_video_folder(video_root=video_root,
                                             save_root=save_root,
                                             video_type=video_type,
                                             frame_interval=frame_interval,
                                             crop_size=crop_size,
                                             scale=scale,
                                             device=device)
    """

elif mode == 'face_labels':
    from utils.face_labels import save_crop_face_from_video_folder


    video_type = ['.mp4', '.avi', '.mov']
    frame_interval = 1

    
    video_root = '/home/wangqiushi/Documents/face_spoofing_datasets/SNTO_test_20191106/split/cellphone_proc/'
    save_root = '/home/wangqiushi/Documents/face_spoofing_datasets/SNTO_test_20191106/crop_face/cellphone/'
    save_crop_face_from_video_folder(video_root=video_root,
                                     save_root=save_root,
                                     video_type=video_type,
                                     frame_interval=frame_interval)
    
    """
    # CASIA-FASD
    root = '/home/wangqiushi/pycode/face_spoofing/datasets/CASIA-FASD/CASIA-FASD/'
    v_sets = {'train_release':20, 'test_release':30}
    save_crop_root = '/home/wangqiushi/pycode/face_spoofing/datasets/CASIA-FASD/crop_face/'

    for v_set in list(v_sets.keys()):
        print('| Set: %s' % v_set)
        for v in range(v_sets[v_set]):
            print('| Sub: %s' % str(v+1))
            video_root = osp.join(root, v_set, str(v+1))
            save_root = osp.join(save_crop_root, v_set, str(v+1))

            save_crop_face_from_video_folder(video_root=video_root,
                                             save_root=save_root,
                                             video_type=video_type,
                                             frame_interval=frame_interval)
    """