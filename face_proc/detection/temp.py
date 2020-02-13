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


label_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/crop_nb.txt'
image_root = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/Images/'

out_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/crop_nb_clean.txt'
out_root = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/out_root/'

images, labels = get_detection_image_label_list(label_file)

save_item = []

for i, label in enumerate(labels):
    image_path = os.path.join(image_root, images[i])
    _, image_file = os.path.split(image_path)
    img = cv2.imread(image_path)
    height, width, channel = img.shape

    ids = []
    bbox = []
    for lb in label:
        lb_list = list(map(int, lb.split()))
        ids.append(lb_list[0])
        bbox.append(lb_list[1:])

    box_in = True
    for j, b in enumerate(bbox):
        out_box = [0, 0, width, height]
        box_in = box_in_box(b, out_box) and box_in
    if box_in:
        save_item.append((','.join([image_file, *label])))
    else:
        shutil.move(image_path, out_root)


    if i % 10 == 0 and i > 0:
        print('| Reset [%d/%d] Bboxes' % (i, len(labels)))
        print(image_file)
print('-----------------------------------')
with open(out_file, 'w+') as f:
    for s in save_item:
        f.write('%s\n' % s)


