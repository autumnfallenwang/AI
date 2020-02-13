from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import logging
import random
import numpy as np
import PIL
import labelme
import matplotlib.pyplot as plt
import mxnet as mx
import gluoncv as gcv
import gluoncv.data as gdata
import gluoncv.utils as gutils
from mxnet.gluon import nn
from mxnet.gluon.data import Dataset
from gluoncv.data.base import VisionDataset
from .params import MODEL_NAME, PRETRAIN, LOAD_MODEL, LOAD_MODEL_PATH
from .gluoncv_utils_export_helper import export_block
from .vgg import vgg18


def get_root_list(root):
    files = []
    for root, dirs, file in os.walk(root):  
    #    print(root) #当前目录路径 
    #    print(dirs) #当前路径下所有子目录
    #    print(file)
        files.append(file)
    return files[0]


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
            lbs[-1] = lbs[-1].replace('\n', '')
            labels.append(lbs)
    if shuffle:
        _shuffle_two_list(images, labels, 123)
    return images, labels


class ImageLabelDataset(Dataset):
    def __init__(self, image_root, label_file, shuffle=True, transform=None):
        self._image_root = image_root
        self._label_file = label_file
        self._shuffle = shuffle
        self._transform = transform
        self.images, self.labels = get_image_label_list(label_file=label_file, shuffle=shuffle)
        self.image_num = len(self.images)
        self.classes = sorted(list(set(self.labels)))
        self.class_num = len(self.classes)
        print("| Load %d images and labels" % self.image_num)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        img = mx.image.imread(os.path.join(self._image_root, image_path))
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return self.image_num


class DetectionDataset(VisionDataset):
    def __init__(self, image_root, label_file, classes, map_file=None, 
                 shuffle=True, transform=None, validate_label=False):
        super(DetectionDataset, self).__init__(image_root)
        self._image_root = os.path.expanduser(image_root)
        self._label_file = os.path.expanduser(label_file)
        self.CLASSES = classes
        self._shuffle = shuffle
        self._transform = transform
        self.validate_label = validate_label
        self.images, self.labels = get_detection_image_label_list(label_file=label_file, shuffle=shuffle)
        self.image_num = len(self.images)
        self.class_num = self.num_class
        if map_file:
            self.index_map = self._get_map_dict(map_file)
            # Increase: classes match validate
        else:
            self.index_map = dict(zip(classes, range(self.num_class)))
        self.parse_labels = self._load_labels()
        print("| Load %d images and labels" % self.image_num)

    @property
    def classes(self):
        """Category names."""
        return self.CLASSES

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.parse_labels[idx]
        img = mx.image.imread(os.path.join(self._image_root, image))
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _get_map_dict(self, map_file):
        map_from, map_to = get_image_label_list(map_file, False)
        map_dict = dict(zip(map_from, map_to))
        return map_dict

    def _load_labels(self):
        lbs = self.labels
        labels_parser = []
        for i, lb in enumerate(lbs):
            lb_parser = []
            if self.validate_label:
                img = mx.image.imread(os.path.join(self._image_root, self.images[i]))
                height, width, _ = img.shape

            for item in lb:
                it = item.split()                
                id_orig = it[0]
                id_map = self.index_map[id_orig]                
                box = list(map(int, it[1:]))
                difficult = 0.0

                if self.validate_label:
                    try:
                        self._validate_label(*box, width, height)
                    except AssertionError as e:
                        raise RuntimeError("Invalid label at {}, {}".format(self.images[i], e))
                lb_parser.append([*box, id_map, difficult])

            labels_parser.append(np.array(lb_parser))
        return labels_parser

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)


def get_model(class_num, 
              ctx, 
              model_name=MODEL_NAME, 
              pretrain=PRETRAIN, 
              load_model=LOAD_MODEL, 
              load_model_path=LOAD_MODEL_PATH):

    last_fc = ['resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d']
    last_output = ['resnet50_v2', 'densenet169', 'densenet201', 'senet_154', 'vgg16', 'vgg18']

    if model_name == 'vgg18':
        finetune_net = vgg18(pretrained=pretrain)
    else:    
        finetune_net = gcv.model_zoo.get_model(model_name, pretrained=pretrain)

    if pretrain == False:
        finetune_net.initialize(mx.init.MSRAPrelu())

    if model_name in last_fc:
        with finetune_net.name_scope():
            finetune_net.fc = nn.Dense(class_num)
        finetune_net.fc.initialize(mx.init.Xavier())
    elif model_name in last_output:
        with finetune_net.name_scope():
            finetune_net.output = nn.Dense(class_num)
        finetune_net.output.initialize(mx.init.Xavier())

    if load_model:
        finetune_net.load_parameters(load_model_path)

    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    print(finetune_net)

    return finetune_net


def create_image_folder_from_label_file(image_root, label_file, save_root):
    print('INFO')
    print('-' * 80)
    print('| Image Root: %s' % image_root)
    print('| Label File: %s' % label_file)
    print('| Save Root: %s' % save_root)

    os.makedirs(save_root, exist_ok=True)
    image_list, label_list = get_image_label_list(label_file, False)
    classes = sorted(list(set(label_list)))
    print('| Copy Images Num: %d' % len(image_list))
    print('| Classes: %s' % classes)
    for c in classes:
        class_image_root = os.path.join(save_root, str(c))
        os.makedirs(class_image_root, exist_ok=True)
        print('| %s: \t%d' % (str(c), label_list.count(c)))

    print()
    print('Copying...')
    print('-' * 80)
    for i in range(0, len(image_list)):
        image = image_list[i]
        label = label_list[i]
        class_image_root = os.path.join(save_root, str(label))
        image_path = os.path.join(image_root, image)
        image_save_path = os.path.join(class_image_root, image)
        shutil.copy(image_path, image_save_path)
        if i % 1000 == 0 and i > 0:
            print('| Copied [%d/%d] Images' % (i, len(image_list)))


def export_model(model_name, model_path, class_num, save_root='./'):
    print('INFO')
    print('-' * 80)
    print('| Model Name: %s' % model_name)
    print('| Model Path: %s' % model_path)
    print('| Class Num: %s' % class_num)
    print('| Save Root: %s' % save_root)
    print()
    print('Exporting...')
    print('-' * 80)

    ctx = mx.cpu()
    model = get_model(class_num=class_num, 
                      ctx=ctx, 
                      model_name=model_name, 
                      pretrain=False, 
                      load_model=True, 
                      load_model_path=model_path)

    _, file_name = os.path.split(model_path)
    file_name, _ = os.path.splitext(file_name)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, file_name)

    export_block(save_path, model, preprocess=True, layout='CHW')
    print('| Save Model in %s' % save_path)


def export_detection_model(model_name, model_path, model_classes, save_root='./'):
    print('INFO')
    print('-' * 80)
    print('| Model Name: %s' % model_name)
    print('| Model Path: %s' % model_path)
    print('| Model Classes: %s' % str(model_classes))
    print('| Save Root: %s' % save_root)
    print()
    print('Exporting...')
    print('-' * 80)

    model = gcv.model_zoo.get_model(model_name, pretrained_base=True)
    if model_name.startswith('ssd'):
        model.reset_class(model_classes)
    model.load_parameters(model_path)

    _, file_name = os.path.split(model_path)
    file_name, _ = os.path.splitext(file_name)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, file_name)

    gutils.export_block(save_path, model, preprocess=False, layout='CHW')
    print('| Save Model in %s' % save_path)


def export_segmentation_model(model_name, backbone, dataset, model_path, save_root='./'):
    print('INFO')
    print('-' * 80)
    print('| Model Name: %s' % model_name)
    print('| Backbone: %s' % backbone)
    print('| Dataset: %s' % dataset)
    print('| Model Path: %s' % model_path)
    print('| Save Root: %s' % save_root)
    print()
    print('Exporting...')
    print('-' * 80)

    model = gcv.model_zoo.segbase.get_segmentation_model(model=model_name, 
                                                         dataset=dataset, 
                                                         backbone=backbone, 
                                                         aux=True)
    #                                                     crop_size=480)
    model.load_parameters(model_path)

    _, file_name = os.path.split(model_path)
    file_name, _ = os.path.splitext(file_name)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, file_name)

    gutils.export_block(save_path, model, preprocess=False, layout='CHW')
    print('| Save Model in %s' % save_path)


def plot_bbox_with_symbol_model(image_root, model_path, 
                                save_root='./bboxes/', 
                                threshold=0.5, 
                                classes=None, 
                                ctx=mx.cpu()):
    print('INFO')
    print('-' * 80)
    print('| Image Root: %s' % image_root)
    print('| Model Path: %s' % model_path)
    print('| Classes: %s' % str(classes))
    print('| Save Root: %s' % save_root)
    print('| Threshold: %.4f' % threshold)
    print('| Context: %s' % ctx)
    print('Exporting...')
    print('-' * 80)

    os.makedirs(save_root, exist_ok=True)
    images = get_root_list(image_root)
    model = nn.SymbolBlock.imports(model_path+'-symbol.json', ['data'], 
                                   model_path+'-0000.params', ctx=ctx)

    print()
    print('Plotting...')
    print('-' * 80)
    for i, image in enumerate(images):
        image_path = os.path.join(image_root, image)
        image_bbox_path = os.path.join(save_root, image)

        x, img = gdata.transforms.presets.ssd.load_test(image_path, short=512)
        x = x.copyto(ctx)

        class_IDs, scores, bounding_boxs = model(x)
        if classes is not None:
            ax = gutils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                      class_IDs[0], thresh=threshold, class_names=classes)
        else:
            ax = gutils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                      class_IDs[0], thresh=threshold, class_names=model.classes)
        plt.savefig(image_bbox_path)
        plt.cla()
        plt.clf()
        plt.close()

        if i % 10 == 0 and i > 0:
            print('| Ploted [%d/%d] Bboxes' % (i, len(images)))


def plot_mask_with_gluon_model(image_root, model_path, 
                               model_name, 
                               backbone, 
                               dataset, 
                               label_file, 
                               save_root='./mask/', 
                               ctx=mx.cpu()):
    print('INFO')
    print('-' * 80)
    print('| Image Root: %s' % image_root)
    print('| Model Path: %s' % model_path)
    print('| Model Name: %s' % model_name)
    print('| Backbone: %s' % backbone)
    print('| Dataset: %s' % dataset)
    print('| Label File: %s' % label_file)
    print('| Save Root: %s' % save_root)
    print('| Context: %s' % ctx)
    print('Exporting...')
    print('-' * 80)

    os.makedirs(save_root, exist_ok=True)
    model = gcv.model_zoo.segbase.get_segmentation_model(model=model_name, 
                                                         dataset=dataset, 
                                                         backbone=backbone, 
                                                         aux=True)
    #                                                     crop_size=480)
    model.load_parameters(model_path, ctx=ctx)

    images = get_root_list(image_root)

    print()
    print('Plotting...')
    print('-' * 80)
    for i, image in enumerate(images):
        image_path = os.path.join(image_root, image)
        image_mask_path = os.path.join(save_root, image.replace('.jpg', '.png'))
        image_on_mask_path = os.path.join(save_root, image)

        img = mx.image.imread(image_path)

        transform_fn = mx.gluon.data.vision.transforms.Compose([
            mx.gluon.data.vision.transforms.ToTensor(),
            mx.gluon.data.vision.transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        img = transform_fn(img)
        img = img.expand_dims(0).as_in_context(ctx)
        

        output = model.demo(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        # mask = gutils.viz.get_color_pallete(predict, 'pascal_voc')
        # mask.save(image_mask_path)

        class_names = []
        class_name_to_id = {}
        for index, line in enumerate(open(label_file).readlines()):
            class_id = index - 1  # starts with -1
            class_name = line.strip()
            class_name_to_id[class_name] = class_id
            if class_id == -1:
                assert class_name == '__ignore__'
                continue
            elif class_id == 0:
                assert class_name == '_background_'
            class_names.append(class_name)
        class_names = tuple(class_names)

        colormap = labelme.utils.label_colormap(255)

        lbl = predict.astype('uint8')
        img_orig = np.asarray(PIL.Image.open(image_path))
        viz = labelme.utils.draw_label(lbl, img_orig, class_names, colormap=colormap)

        PIL.Image.fromarray(viz).save(image_on_mask_path)
        
        if i % 10 == 0 and i > 0:
            print('| Ploted [%d/%d] Maskes' % (i, len(images)))


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


def _detection_lst_write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    labels[:, 1:][labels[:, 1:]>1.0] = 1.0
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

def create_detection_lst_from_label_file(image_root, label_file, map_file, save_root='./'):
    print('INFO')
    print('-' * 80)
    print('| Image Root: %s' % image_root)
    print('| Label File: %s' % label_file)
    print('| Map File: %s' % map_file)
    print()

    images, lbs = get_detection_image_label_list(label_file, False)
    
    map_from, map_to = get_image_label_list(map_file, False)
    map_from = list(map(int, map_from))
    map_list = list(zip(map_from, map_to))

    _, save_file = os.path.split(label_file)
    save_file = save_file.replace('.txt', '.lst')
    save_path = os.path.join(save_root, save_file)

    print('Saving...')
    print('-' * 80)
    with open(save_path, 'w') as fw:
        for i, lb in enumerate(lbs):
            img = mx.image.imread(os.path.join(image_root, images[i]))
            ids = []
            boxes = []
            for item in lb:
                it = list(map(int, item.split()))
                boxes.append(it[1:])
                for m in map_list:
                    if it[0] == m[0]:
                        ids.append(m[1])

            ids = np.array(ids)
            boxes = np.array(boxes)

            line = _detection_lst_write_line(images[i], img.shape, boxes, ids, i)
            fw.write(line)

            if i % 1000 == 0 and i > 0:
                print('| Saved [%d/%d] items' % (i, len(lbs)))
    
    print()
    print('| Save Lst: %s' % save_path)


class SingleMetrics(object):
    def __init__(self, classes):
        self.classes = classes
        class_num = len(classes)
        self.confusion_matrix = np.zeros((class_num, class_num), dtype=int)
        self.metrics = np.zeros((class_num, 4), dtype=int) # [TP, FP, TN, FN]

    def get_single_cf_matrix(self, pred, label):
        self.confusion_matrix[label, pred] += 1

    def get_metrics(self):
        for cl in self.classes:
            TP = self.confusion_matrix[cl, cl]
            FP = self.confusion_matrix.sum(axis=0)[cl] - TP
            FN = self.confusion_matrix.sum(axis=1)[cl] - TP
            TN = self.confusion_matrix.sum() - TP - FP - FN
            self.metrics[cl] = [TP, FP, TN, FN]

    def print_metrics(self):
        for cl in self.classes:
            TP = self.metrics[cl, 0]
            FP = self.metrics[cl, 1]
            TN = self.metrics[cl, 2]
            FN = self.metrics[cl, 3]
            print('| Label %s:' % cl)
            print('| TP FP TN FN: %s' % self.metrics[cl])
            if TP:
                precision = TP/(TP + FP)
                recall = TP/(TP + FN)
                accuracy = (TP + TN)/(TP + TN + FP + FN)
                
                sensitivity = TP/(TP + FN)
                specificity = TN/(TN + FP)
                
                print('| Precision = %.4f, Recall = %.4f, Accuracy = %.4f' % (precision, recall, accuracy))
                print('| Sensitivity = %.4f, Specificity = %.4f' % (sensitivity, specificity))
                print('|')

    def print_cf_matrix(self):
        print('| Confusion matrix:')
        print('   --->  predict')
        print('   |')
        print('  \|/')
        print(' label')
        print()
        print(self.confusion_matrix)
        print('|')


