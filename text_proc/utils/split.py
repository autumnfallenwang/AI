import os
import os.path as osp
import sys

from .io import *


def get_train_valid_test_split(label_file, attribute, shuffle, 
                               split_rate=[0.8, 0.1, 0.1], select_rate=1,
                               save_path=['./train.txt', './valid.txt', './test.txt']):
    # attribute in ['cls', 'det']
    
    print('INFO')
    print('-' * 80)
    print('| Label File: %s' % label_file)
    print('| Attribute: %s' % attribute)
    print('| Shuffle: %s' % shuffle)
    print('| Split Rate: %s' % str(split_rate))
    print('| Select Rate: %s' % select_rate)
    print()

    images = []
    labels = []
    lbs = []
    if attribute.lower() == 'cls':
        split = ' '
        images, labels = get_image_label_list(label_file, shuffle, split=split)
    elif attribute.lower() == 'det':
        split = ','
        images, lbs = get_detection_image_label_list(label_file, shuffle, split)
        for lb in lbs:
            labels.append(int(lb[0].split()[0]))

    classes = list(set(labels))
    class_item_num = []
    class_split_num = []
    c_count = []
    
    print('Spliting Dataset...')
    print('-' * 80)
    print('| Classes: %s' % str(classes))
    for c in classes:
        class_item_num.append(labels.count(c))
    print('| class: \ttrain \tvalid \ttest')
    for c_i in range(len(classes)):
        c_train = int(split_rate[0] * class_item_num[c_i] * select_rate)
        c_valid = int(split_rate[1] * class_item_num[c_i] * select_rate)
        c_test = int(split_rate[2] * class_item_num[c_i] * select_rate)
        class_split_num.append([c_train, c_train+c_valid, c_train+c_valid+c_test])
        c_count.append(0)
        print('| '+str(classes[c_i])+': \t\t'+str(c_train)+'\t'+str(c_valid)+'\t'+str(c_test))

    images_train = []
    labels_train = []
    images_valid = []
    labels_valid = []    
    images_test = []
    labels_test = []

    for i in range(len(labels)):
        for c_i, c in enumerate(classes):
            if labels[i] == c:
                if attribute.lower() == 'cls':
                    lb = labels[i]
                elif attribute.lower() == 'det':
                    lb = ','.join(lbs[i])

                if 0 <= c_count[c_i] < class_split_num[c_i][0]:
                    images_train.append(images[i])
                    labels_train.append(lb)
                elif class_split_num[c_i][0] <= c_count[c_i] < class_split_num[c_i][1]:
                    images_valid.append(images[i])
                    labels_valid.append(lb)
                elif class_split_num[c_i][1] <= c_count[c_i] < class_split_num[c_i][2]:
                    images_test.append(images[i])
                    labels_test.append(lb)
                c_count[c_i] += 1
    print()
    print('Saving...')
    print('-' * 80)

    with open(save_path[0], 'w+') as f:
        for i in range(len(labels_train)):
            f.write('%s%s%s\n' % (images_train[i], split, labels_train[i]))
        print('| Save Train: %s' % save_path[0])
    with open(save_path[1], 'w+') as f:
        for i in range(len(labels_valid)):
            f.write('%s%s%s\n' % (images_valid[i], split, labels_valid[i]))
        print('| Save Valid: %s' % save_path[1])
    with open(save_path[2], 'w+') as f:
        for i in range(len(labels_test)):
            f.write('%s%s%s\n' % (images_test[i], split, labels_test[i]))
        print('| Save Test: %s' % save_path[2])