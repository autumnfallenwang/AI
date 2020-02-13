from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image, ImageStat
sys.path.append('../../')
from models import alexnet, densenet, inception, squeezenet, vgg, resnet
from .params import FINETUNE, MODEL_TYPE, MODEL_DEPTH_OR_VERSION


def PIL_image_loader(path):
    return Image.open(path).convert('RGB')

def _shuffle_two_list(a, b, seed):
    if sys.version_info.major == 2:
        zip_list = zip(a, b)
    else:
        zip_list = list(zip(a, b))
    random.seed(seed)
    random.shuffle(zip_list)
    a[:], b[:] = zip(*zip_list)

def _get_image_label_list(label_file, shuffle, split=' '):
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

class ImageData(Dataset):
    def __init__(self, image_dir, label_file, shuffle=True, transform=None, image_loader=PIL_image_loader):
        self.image_dir = image_dir
        self.label_file = label_file
        self.shuffle = shuffle
        self.transform = transform
        self.image_loader = image_loader
        self.images, self.labels = _get_image_label_list(label_file=label_file, shuffle=shuffle)
        self.image_num = len(self.images)
        self.classes = sorted(list(set(self.labels)))
        self.class_num = len(self.classes)
        print("| Load %d images and labels" % self.image_num)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = self.image_loader(os.path.join(self.image_dir, image_path))
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
        
    def __len__(self):
        return self.image_num


def get_model(class_num):
    if (MODEL_TYPE == 'alexnet'):
        model = alexnet.alexnet(pretrained=FINETUNE)
    elif (MODEL_TYPE == 'vgg'):
        if(MODEL_DEPTH_OR_VERSION == 11):
            model = vgg.vgg11(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 13):
            model = vgg.vgg13(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 16):
            model = vgg.vgg16(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 19):
            model = vgg.vgg19(pretrained=FINETUNE)
        else:
            print('Error : VGG should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
    elif (MODEL_TYPE == 'squeezenet'):
        if(MODEL_DEPTH_OR_VERSION == 0 or MODEL_DEPTH_OR_VERSION == 'v0'):
            model = squeezenet.squeezenet1_0(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 1 or MODEL_DEPTH_OR_VERSION == 'v1'):
            model = squeezenet.squeezenet1_1(pretrained=FINETUNE)
        else:
            print('Error : Squeezenet should have version of either [0, 1]')
            sys.exit(1)
    elif (MODEL_TYPE == 'resnet'):
        if(MODEL_DEPTH_OR_VERSION == 18):
            model = resnet.resnet18(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 34):
            model = resnet.resnet34(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 50):
            model = resnet.resnet50(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 101):
            model = resnet.resnet101(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 152):
            model = resnet.resnet152(pretrained=FINETUNE)
        else:
            print('Error : Resnet should have depth of either [18, 34, 50, 101, 152]')
            sys.exit(1)
    elif (MODEL_TYPE == 'densenet'):
        if(MODEL_DEPTH_OR_VERSION == 121):
            model = densenet.densenet121(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 169):
            model = densenet.densenet169(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 161):
            model = densenet.densenet161(pretrained=FINETUNE)
        elif(MODEL_DEPTH_OR_VERSION == 201):
            model = densenet.densenet201(pretrained=FINETUNE)
        else:
            print('Error : Densenet should have depth of either [121, 169, 161, 201]')
            sys.exit(1)
    elif (MODEL_TYPE == 'inception'):
        if(MODEL_DEPTH_OR_VERSION == 3 or MODEL_DEPTH_OR_VERSION == 'v3'):
            model = inception.inception_v3(pretrained=FINETUNE)
        else:
            print('Error : Inception should have version of either [3, ]')
            sys.exit(1)
    else:
        print('Error : Network should be either [alexnet / squeezenet / vgg / resnet / densenet / inception]')
        sys.exit(1)

    if(MODEL_TYPE == 'alexnet' or MODEL_TYPE == 'vgg'):
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, class_num))
        model.classifier = nn.Sequential(*feature_model)
    elif(MODEL_TYPE == 'resnet' or MODEL_TYPE == 'inception'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)
    elif(MODEL_TYPE == 'densenet'):
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, class_num)

    return model


def get_parallel_model(class_num, multi_gpu, gpu_list, device):
    model = get_model(class_num)
    if multi_gpu:
        model = nn.DataParallel(model, gpu_list) # device_ids
    # CUDA
    model.to(device)
    return model

def get_train_valid_test_split(label_file, shuffle, split_rate=[0.8, 0.1, 0.1], select_rate=1,
                               save_path=['./train.txt', './valid.txt', './test.txt']):
    print('INFO')
    print('-' * 80)
    print('| Label File: %s' % label_file)
    print('| Shuffle: %s' % shuffle)
    print('| Split Rate: %s' % str(split_rate))
    print('| Select Rate: %s' % select_rate)
    print()
    images, labels = _get_image_label_list(label_file, shuffle)
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
                if 0 <= c_count[c_i] < class_split_num[c_i][0]:
                    images_train.append(images[i])
                    labels_train.append(labels[i])
                elif class_split_num[c_i][0] <= c_count[c_i] < class_split_num[c_i][1]:
                    images_valid.append(images[i])
                    labels_valid.append(labels[i])
                elif class_split_num[c_i][1] <= c_count[c_i] < class_split_num[c_i][2]:
                    images_test.append(images[i])
                    labels_test.append(labels[i])
                c_count[c_i] += 1
    print()
    print('Saving...')
    print('-' * 80)

    with open(save_path[0], 'w+') as f:
        for i in range(len(labels_train)):
            f.write('%s %s\n' % (images_train[i], str(labels_train[i])))
        print('| Save Train: %s' % save_path[0])
    with open(save_path[1], 'w+') as f:
        for i in range(len(labels_valid)):
            f.write('%s %s\n' % (images_valid[i], str(labels_valid[i])))
        print('| Save Valid: %s' % save_path[1])
    with open(save_path[2], 'w+') as f:
        for i in range(len(labels_test)):
            f.write('%s %s\n' % (images_test[i], str(labels_test[i])))
        print('| Save Test: %s' % save_path[2])



def resize_image(image_list, image_dir, save_dir, resize_size=(256, 256)):
    # Size 264 allows centre-crop of 224 for image-augmentation
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, image_path in enumerate(image_list):
        _, image_name = os.path.split(image_path)
        image_path = os.path.join(image_dir, image_path)
        image = Image.open(image_path).resize(resize_size, Image.BILINEAR).convert('RGB')
        image.save(os.path.join(save_dir, image_name))
        if i % 1000 == 0 and i > 0:
            print('| Resized [%d/%d] Images' % (i, len(image_list)))

def resize_image_from_label_file(label_file, image_dir, save_dir, resize_size=(256, 256)):
    print('INFO')
    print('-' * 80)
    print('| Label File: %s' % label_file)
    print('| Image Dir: %s' % image_dir)
    print('| Save Dir: %s' % save_dir)
    print('| Resize Size: %s' % str(resize_size))
    image_list, _ = _get_image_label_list(label_file, False)
    image_num = len(image_list)
    print('| Resize Images Num: %d' % image_num)
    print()

    print('Resizing...')
    print('-' * 80)
    resize_image(image_list=image_list, image_dir=image_dir, save_dir=save_dir, resize_size=resize_size)


def cal_image_mean_std(image_list, image_dir):
    r_array = np.array([])
    g_array = np.array([])
    b_array = np.array([])

    for i, image_path in enumerate(image_list):
        image_path = os.path.join(image_dir, image_path)
        image = Image.open(image_path)
        stat = ImageStat.Stat(image)

        r_array = np.append(r_array, stat.mean[0]/255)
        g_array = np.append(g_array, stat.mean[1]/255)
        b_array = np.append(b_array, stat.mean[2]/255)
        if i % 1000 == 0 and i > 0:
            print('| Calculated [%d/%d] Images' % (i, len(image_list)))

    mean = [r_array.mean(), g_array.mean(), b_array.mean()]
    std = [r_array.std(), g_array.std(), b_array.std()]
    return mean, std

def cal_image_stat_from_label_file(label_file, image_dir):
    print('INFO')
    print('-' * 80)
    print('| Label File: %s' % label_file)
    print('| Image Dir: %s' % image_dir)
    image_list, _ = _get_image_label_list(label_file, False)
    image_num = len(image_list)
    print('| Calculate Images Num: %d' % image_num)
    print()
    print('Calculating...')
    print('-' * 80)
    mean, std = cal_image_mean_std(image_list=image_list, image_dir=image_dir)
    print()
    print('| Mean: %s' % str(mean))
    print('| Std: %s' % str(std))

def convert_model_to_onnx(model_path, class_num, multi_gpu, gpu_list, save_dir='./'):
    print('INFO')
    print('-' * 80)
    print('| Model Path: %s' % model_path)
    print('| Save Dir: %s' % save_dir)
    print('| Class Num: %s' % class_num)
    print('| Multi_GPU: %s' % multi_gpu)
    print('| GPU List: %s' % str(gpu_list))
    print()
    print('Converting')
    print('-' * 80)
    device0 = torch.device("cuda:0")
    device_cpu = torch.device('cpu')
    state_dict = torch.load(model_path)
    model = get_parallel_model(class_num, multi_gpu, gpu_list, device0)
    model.load_state_dict(state_dict)

    if multi_gpu:
        model = model.module

    #model.to(device_cpu)
    _, model_name = os.path.split(model_path)
    model_name, _ = os.path.splitext(model_name)
    filename = os.path.join(save_dir, model_name+'.onnx')
    print('| Save Model in %s' % filename)
    dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()
    torch.onnx.export(model, dummy_input, filename, verbose=True)


class BatchMetrics(object):
    def __init__(self, classes):
        self.classes = classes
        class_num = len(classes)
        self.confusion_matrix = np.zeros((class_num, class_num), dtype=int)
        self.metrics = np.zeros((class_num, 4), dtype=int) # [TP, FP, TN, FN]

    def get_batch_cf_matrix(self, pred_list, label_list):
        for i in range(len(pred_list)):
            pred = pred_list[i]
            label = label_list[i]
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
                TPR = TP/(TP + FN)
                FPR = FP/(FP + TN)
                print('| Precision = %.4f, Recall = %.4f, Accuracy = %.4f' % (precision, recall, accuracy))
                print('| 1 - P = %.4f, 1 - R = %.4f' % (1-precision, 1-recall))
                print('| TPR = %.4f, FPR = %.4f' % (TPR, FPR))
                print('|')


class TrainLossAcc(object):
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.lr = []

    def get_epoch_value(self, epoch_loss, epoch_acc, lr, phase):
        if phase == 'train':
            self.train_loss.append(epoch_loss)
            self.train_acc.append(epoch_acc)
            self.lr.append(lr)
        else:
            self.valid_loss.append(epoch_loss)
            self.valid_acc.append(epoch_acc)
    
    def save_log(self, log_path):
        with open(log_path, 'w+') as f:
            for i in range(len(self.train_loss)):
                f.write('%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f\n' % 
                    (i+1,
                     self.train_loss[i],
                     self.valid_loss[i],
                     self.train_acc[i],
                     self.valid_acc[i],
                     self.lr[i]))

    def read_log(self, log_path):
        train_loss = []
        train_acc = []
        valid_loss = []
        valid_acc = []
        lr = []
        with open(log_path, 'r') as read_file:
            while True:
                lines = read_file.readline()
                if not lines:
                    break
                string = lines.split('\t\t')
                train_loss.append(float(string[1]))
                valid_loss.append(float(string[2]))
                train_acc.append(float(string[3]))
                valid_acc.append(float(string[4]))
                lr.append(float(string[5]))
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.train_acc = train_acc
        self.valid_acc = valid_acc
        self.lr = lr
