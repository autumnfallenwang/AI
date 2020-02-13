import os
import shutil
import numpy as np
import cv2


def box_in_box(in_box, out_box):
    xmin = in_box[0]>out_box[0] and in_box[0]<out_box[2]
    ymin = in_box[1]>out_box[1] and in_box[1]<out_box[3]
    xmax = in_box[2]>out_box[0] and in_box[2]<out_box[2]
    ymax = in_box[3]>out_box[1] and in_box[3]<out_box[3]
    return xmin and ymin and xmax and ymax

def reset_box(in_box, out_box):
    safe = 0
    new_box = [in_box[0]-out_box[0], 
               in_box[1]-out_box[1],
               in_box[2]-out_box[0],
               in_box[3]-out_box[1]]
    if box_in_box(in_box, out_box):
        safe = 1
    return safe, new_box


def get_detection_image_label_list(label_file, split=','):
    images = []
    labels = []
    with open(label_file, 'r') as read_file:
        while True:
            lines = read_file.readline()
            if not lines:
                break
            string = lines.split(split)
            images.append(string[0])
            lbs = string[1:]
            lbs[-1] = lbs[-1].replace('\n', '')
            labels.append(lbs)
    return images, labels


label_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/imageInfo_xr.txt'
image_root = '/'
crop_box_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/crop_box.txt'
crop_root = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/xr_crop/'

nb_label_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/crop_nb_crop.txt'
bad_box_root = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/bad_box/'

images, labels = get_detection_image_label_list(label_file)
images_c, crop_box = get_detection_image_label_list(crop_box_file)

save_item = []

for i, label in enumerate(labels):
    image_path = os.path.join(image_root, images[i])
    _, image_file = os.path.split(image_path)

    if image_file in images_c:
        ids = []
        bbox = []
        for lb in label:
            lb_list = list(map(int, lb.split()))
            ids.append(lb_list[0])
            bbox.append(lb_list[1:])

        nlabel = []
        for j, b in enumerate(bbox):
            out_box = crop_box[images_c.index(image_file)]
            out_box = list(map(int, out_box[0].split()))
            safe, nb = reset_box(b, out_box)
            if safe:
                item = [ids[j], *nb]
                nlabel.append(' '.join(list(map(str, item))))

        if nlabel != []:
            save_item.append((','.join([image_file, *nlabel])))
#        else:
#            shutil.copy(os.path.join(crop_root, image_file), bad_box_root)

        if i % 10 == 0 and i > 0:
            print('| Reset [%d/%d] Bboxes' % (i, len(labels)))
            print(image_file)

#    else:
#        save_item.append((','.join([image_file, *label])))

with open(nb_label_file, 'w+') as f:
    for s in save_item:
        f.write('%s\n' % s)

















