import os
import shutil
import numpy as np
import cv2


def mul_or(l):
    m = 0
    for i in l:
        m = m or i
    return m

def save_zone(img, center, lenth):
    h, w = img.shape
    x0, y0 = center
    assert x0>=0 and x0<=w and y0>=0 and y0<=h
    
    xmin = x0-lenth
    xmax = x0+lenth
    ymin = y0-lenth
    ymax = y0+lenth
    if xmin < 0:
        xmin=0
    if xmax > w:
        xmax = w
    if ymin<0:
        ymin=0
    if ymax>h:
        ymax=h
    return [xmin, ymin, xmax, ymax]

def center_in_dark(img, center, dark_thres=40, box_lenth=2):
    box = save_zone(img, center, box_lenth)
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
test_root = '/mnt/public_data/wangqiushi/DR5_bbox/test/'

label_map = {10:0, 20:1, 30:2, 40:3, 50:4}
class_map = {0:'jmqz', 1:'ky', 2:'ml', 3:'xr', 4:'xsw'}
class_names = ('jmqz', 'ky', 'ml', 'xr', 'xsw')

images, labels = get_detection_image_label_list(label_file)


for i, label in enumerate(labels):
    image_path = os.path.join(image_root, images[i])
    _, image_file = os.path.split(image_path)

    ids = []
    bbox = []
    for lb in label:
        lb_list = list(map(int, lb.split()))
        ids.append(label_map[lb_list[0]])
        bbox.append(lb_list[1:])

    j = 0
    if ids[0] == 4:
        img = cv2.imread(image_path,cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corner = corner_in_dark(gray, bbox[0])
        
        image_name = os.path.splitext(image_file)[0]
        image_name = image_name+'_dark_'+'_'.join(list(map(str,corner)))+'.jpg'

        image_bbox_path = os.path.join(save_root, 'xsw', image_file)
        image_test_save_path = os.path.join(test_root, image_name)
        print(image_test_save_path)
        shutil.copy(image_bbox_path, image_test_save_path)

        j+=1

    if j == 30:
        break

