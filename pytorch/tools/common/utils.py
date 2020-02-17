import os
import os.path as osp
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
from torch.utils.data import Dataset
from PIL import Image, ImageStat
from torchvision import models

sys.path.append('../../')
from text_proc.utils.label_class import ClsLabel


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ClsDataset(Dataset):
    def __init__(self, image_root, label_path, shuffle=True, transform=None, image_loader=default_loader):
        super().__init__()
        self.image_root = image_root
        self.label_path = label_path
        self.shuffle = shuffle
        self.transform = transform
        self.image_loader = image_loader

        cls_label = ClsLabel(label_path)
        self.cls_label = cls_label.shuffle() if shuffle else cls_label
        self.label_num = len(self.cls_label)
        self.classes = self.cls_label.count().index.to_list()
        self.class_num = len(self.classes)
        print("| Load %d Cls Labels" % self.label_num)
        print("| Class Number is %d" % self.class_num)

    def __getitem__(self, idx):
        image_path = self.cls_label.loc[idx, 'image']
        image = self.image_loader(osp.join(self.image_root, image_path))
        if self.transform is not None:
            image = self.transform(image)
        label = self.cls_label.loc[idx, 'label']
        return image, label

    def __len__(self):
        return self.label_num


def get_model(class_num, finetune, model_type, model_name):

    pretrain_str = 'pretrained='+str(finetune)
    model_str = 'models.' + model_name
    model_eval = model_str + '(' + pretrain_str + ')'
    model = eval(model_eval)

    if model_type in ['alexnet', 'vgg']:
        num_ftrs = model.classifier[6].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, class_num))
        model.classifier = nn.Sequential(*feature_model)

    elif model_type in ['resnet', 'inception', 'googlenet', 'shufflenet']:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)

    elif model_type in ['densenet']:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, class_num)

    elif model_type in ['mnasnet', 'mobilenet']:
        num_ftrs = model.classifier[1].in_features
        feature_model = list(model.classifier.children())
        feature_model.pop()
        feature_model.append(nn.Linear(num_ftrs, class_num))
        model.classifier = nn.Sequential(*feature_model)

    return model


def get_parallel_model(class_num, finetune, model_type, model_name, multi_gpu, gpu_list, device):
    model = get_model(class_num, finetune, model_type, model_name)
    if multi_gpu:
        model = nn.DataParallel(model, gpu_list) # device_ids
    # CUDA
    model.to(device)
    return model


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
    image_list, _ = get_image_label_list(label_file, False)
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
    image_list, _ = get_image_label_list(label_file, False)
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

"""
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
                print('| Precision = %.4f, Recall = %.4f, Accuracy = %.4f' % (precision, recall, accuracy))
                print('| 1 - P = %.4f, 1 - R = %.4f' % (1-precision, 1-recall))
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
"""
