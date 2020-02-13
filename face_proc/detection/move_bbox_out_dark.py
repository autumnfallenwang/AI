import os
import shutil
import numpy as np
import cv2


def mul_or(l):
    m = 0
    for i in l:
        m = m or i
    return m

def safe_zone(img, center, lenth):
    h, w = img.shape
    x0, y0 = center
    assert x0>=0 and x0<=w and y0>=0 and y0<=h
    
    xmin = x0-lenth
    xmax = x0+lenth
    ymin = y0-lenth
    ymax = y0+lenth
    if xmin < 0:
        xmin = 0
    if xmax > w:
        xmax = w
    if ymin < 0:
        ymin = 0
    if ymax > h:
        ymax = h
    return [xmin, ymin, xmax, ymax]

def center_in_dark(img, center, dark_thres=50, box_lenth=2):
    box = safe_zone(img, center, box_lenth)
    box_clip = img[box[1]:box[3], box[0]:box[2]]
    return np.mean(box_clip)<dark_thres
#    return np.mean(box_clip)

def corner_in_dark(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    LT = [xmin, ymin]
    RT = [xmax, ymin]
    LB = [xmin, ymax]
    RB = [xmax, ymax]
    corner = [0, 0, 0, 0]
    for i, p in enumerate([LT, RT, LB, RB]):
        if center_in_dark(img, p):
            corner[i] = 1
    return corner

def safe_bbox(bbox, min_length=5):
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    safe = 0
    if w >= min_length and h >= min_length:
        safe = 1
    return safe

def move_bbox(img, bbox, step=5):
    xmin, ymin, xmax, ymax = bbox
    corner = corner_in_dark(img, bbox)
    new_bbox = bbox

    while(mul_or(corner) != 0 and safe_bbox(new_bbox, step)):
        if corner[0] == 1:
            xmin += step
            ymin += step
        elif corner[1] == 1:
            xmax -= step
            ymin += step
        elif corner[2] == 1:
            xmin += step
            ymax -= step
        elif corner[3] == 1:
            xmax -= step
            ymax -= step

        new_bbox = [xmin, ymin, xmax, ymax]
        corner = corner_in_dark(img, new_bbox)

    safe = 1
    if safe_bbox(new_bbox, step) == 0:
        safe = 0
    return safe, new_bbox


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


label_file = '/data02/wangqiushi/datasets/DR5/imageInfo.txt'
image_root = '/'
save_root = '/mnt/public_data/wangqiushi/DR5_bbox/'
nb_label_file = '/mnt/public_data/wangqiushi/DR5_bbox/move_bbox/imageInfo_nb.txt'
not_save_label_file = '/mnt/public_data/wangqiushi/DR5_bbox/move_bbox/imageInfo_not_save.txt'

#label_map = {10:0, 20:1, 30:2, 40:3, 50:4}
#class_map = {0:'jmqz', 1:'ky', 2:'ml', 3:'xr', 4:'xsw'}
#class_names = ('jmqz', 'ky', 'ml', 'xr', 'xsw')

images, labels = get_detection_image_label_list(label_file)

save_item = []
not_save_item = []

for i, label in enumerate(labels):
    image_path = os.path.join(image_root, images[i])
    _, image_file = os.path.split(image_path)

    ids = []
    bbox = []
    for lb in label:
        lb_list = list(map(int, lb.split()))
        ids.append(lb_list[0])
        bbox.append(lb_list[1:])

    img = cv2.imread(image_path,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nlabel = []
    for j, b in enumerate(bbox):
        safe, nb = move_bbox(gray, b)
        if safe:
            item = [ids[j], *nb]
            nlabel.append(' '.join(list(map(str, item))))

    if nlabel != []:
        save_item.append((','.join([image_path, *nlabel])))
    else:
        not_save_item.append((','.join([image_path, *label])))

    if i % 10 == 0 and i > 0:
        print('| Moved [%d/%d] Bboxes' % (i, len(labels)))
        print(image_path)

with open(nb_label_file, 'w+') as f:
        for s in save_item:
            f.write('%s\n' % s)

with open(not_save_label_file, 'w+') as f:
        for s in not_save_item:
            f.write('%s\n' % s)