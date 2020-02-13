import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
import gluoncv.utils as gutils

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

label_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/crop_nb_crop.txt'
image_root = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/xr_crop/'
save_root = '/mnt/public_data/wangqiushi/temp/xr_bbox/'

label_map = {10:0, 20:1, 30:2, 40:3, 50:4}
class_map = {0:'jmqz', 1:'ky', 2:'ml', 3:'xr', 4:'xsw'}
class_names = ('jmqz', 'ky', 'ml', 'xr', 'xsw')

images, labels = get_detection_image_label_list(label_file)

for i, label in enumerate(labels):
    image_path = os.path.join(image_root, images[i])
    _, image_file = os.path.split(image_path)
#    image_bbox_path = os.path.join(save_root, images[i])
    img = mx.image.imread(image_path)

    ids = []
    bbox = []
    scores = []
    for lb in label:
        lb_list = list(map(int, lb.split()))
        ids.append(label_map[lb_list[0]])
        bbox.append(lb_list[1:])
        scores.append(1.0)

    if ids[0] == 3:
        ids = np.array(ids).reshape(len(ids), 1)
        bbox = np.array(bbox)
        scores = np.array(scores).reshape(len(ids), 1)

        image_bbox_path = os.path.join(save_root, image_file)

        ax = gutils.viz.plot_bbox(img, bbox, scores, ids, thresh=0.5, class_names=class_names)

        plt.savefig(image_bbox_path)
        plt.cla()
        plt.clf()
        plt.close()

    if i % 10 == 0 and i > 0:
        print('| Ploted [%d/%d] Bboxes' % (i, len(labels)))
        print(image_path)





