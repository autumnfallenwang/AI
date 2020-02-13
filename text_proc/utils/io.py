import os
import os.path as osp
import sys
import random


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


def get_legal_root_list(root, legal_exts):
    file_list = get_root_list(root)
    return keep_legal_exts(file_list, legal_exts)


def _shuffle_two_list(a, b, seed):
    if sys.version_info.major == 2:
        zip_list = zip(a, b)
    else:
        zip_list = list(zip(a, b))
    random.seed(seed)
    random.shuffle(zip_list)
    a[:], b[:] = zip(*zip_list)


def get_image_label_list(label_file, shuffle, split=' '):
    images = []
    labels = []
    with open(label_file, 'r') as read_file:
        while True:
            lines = read_file.readline()
            if not lines:
                break
            string = lines.split(split)
            images.append(string[0])
            labels.append(int(string[1]))
    if shuffle:
        _shuffle_two_list(images, labels, 123)
    return images, labels


def write_image_label_list(image_list, label_list, file_path, split=' '):
    assert len(image_list) == len(label_list)
    with open(file_path, 'w+') as f:
        for i in range(len(image_list)):
            f.write('%s%s%s\n' % (image_list[i], split, label_list[i]))


def get_detection_image_label_list(label_file, shuffle, split=','):
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
            lbs[-1] = lbs[-1].rstrip('\n')
            labels.append(lbs)
    if shuffle:
        _shuffle_two_list(images, labels, 123)
    return images, labels