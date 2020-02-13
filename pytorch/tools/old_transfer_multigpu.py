from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import copy
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from common.utils_old import get_cuda_version, get_cudnn_version, get_gpu_name
from common.utils import ImageData, BatchMetrics, TrainLossAcc, get_parallel_model
from common.params import *


print('INFO')
print('-' * 80)
print("| OS: ", sys.platform)
print("| Python: ", sys.version)
print("| PyTorch: ", torch.__version__)
print("| Numpy: ", np.__version__)

CPU_COUNT = multiprocessing.cpu_count()
print("| CPUs: ", CPU_COUNT)
print("| GPUs: ", get_gpu_name())
print('| ' + get_cuda_version())
print("| CuDNN Version ", get_cudnn_version())
if MULTI_GPU:
    print("| Use GPUs: %s" % str(GPU_LIST))
else:
    print("| Use GPU: %s" % GPU_DEVICE)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_DEVICE)

# Manually scale to multi-gpu
assert torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
# enables cudnn's auto-tuner
torch.backends.cudnn.benchmark=True
torchvision.set_image_backend('accimage')
print("| Image Backend: %s" % torchvision.get_image_backend())
print()

print('Loading Data...')
print('-' * 80)
normalize = transforms.Normalize(TRAIN_RGB_MEAN,
                                 TRAIN_RGB_SD)
if RESIZE:
    transform = {
        'train': transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize
        ]),
    }
else:
    transform = {
        'train': transforms.Compose([
            transforms.RandomCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize
        ]),
    }

print('| Load Train, Valid, Test Dataset:')
datasets = {
    x : ImageData(image_dir=IMAGE_DIR[x],
                  label_file=LABEL_FILES[x],
                  shuffle=True,
                  transform=transform[x])
    for x in ['train', 'valid', 'test']
}
for x in ['train', 'valid', 'test']:
    print('| '+x+': '+LABEL_FILES[x])
print()

dataloaders = {
    x : DataLoader(dataset=datasets[x],
                   batch_size=BATCHSIZE,
                   shuffle=(x=='train'),
                   num_workers=16,
                   pin_memory=True)
    for x in ['train', 'valid', 'test']
}

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'valid', 'test']}
dataset_classes = datasets['train'].classes
dataset_class_num = datasets['train'].class_num

def init_model(model, lr, lr_step, momentum=0.9, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer = optim.SGD(model.parameters(), 
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
    return optimizer, criterion, scheduler

def test_model(model, dataloader, criterion):
    model.eval()
    print("Testing Model...")
    print('-' * 80)
    print('| Test Batch Size %d' % BATCHSIZE)

    running_loss = 0.0
    running_corrects = 0
    classes = dataloader.dataset.classes
    metrics = BatchMetrics(classes)

    since = time.time()
    # Don't save gradients
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Get samples
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # Forwards
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Loss
            loss = criterion(outputs, labels)
            # Log the loss
            running_loss += loss.item() * inputs.size(0)
            corrects = torch.sum(preds == labels.data)
            running_corrects += corrects

            # Metrics
            pred_list = preds.cpu().numpy()
            label_list = labels.cpu().numpy()
            metrics.get_batch_cf_matrix(pred_list, label_list)

            print('| Iter [%3d/%3d]\t\tLoss %.4f\tAcc %.4f'
                  % (batch_idx+1, (dataset_sizes['test']//BATCHSIZE)+1,
                     loss.item(), corrects.double()/inputs.size(0)))
    
    print("\nTest Results")
    print('-' * 80)
    metrics.get_metrics()
    metrics.print_metrics()
    # Fina loss
    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('| Total\t\t\t\tLoss %.4f\tAcc %.4f' % (epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('| Test Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train_model(model, criterion, optimizer, scheduler, epochs):
    print('Training Model...')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_acc = TrainLossAcc()

    os.makedirs(SAVED_MODELS, exist_ok=True)
    os.makedirs(SAVED_LOGS, exist_ok=True)

    time_filename = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model_filename = os.path.join(SAVED_MODELS, SAVED_PREFIX+time_filename+'.pth')
    log_filename = os.path.join(SAVED_LOGS, SAVED_PREFIX+time_filename+'.txt')

    for epoch in range(epochs):
        since = time.time()
        # print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 80)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()   # warning in pytorch 1.3
                lr = optimizer.param_groups[0]['lr']
                print('| Train Epoch #%d, LR=%f' %(epoch+1, lr))
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            tot = 0

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                tot += labels.size(0)

                if phase == 'train':
                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\t\tLoss %.4f\tAcc %.4f'
                                     % (epoch+1, epochs,
                                        batch_idx+1, (dataset_sizes[phase]//BATCHSIZE)+1,
                                        loss.item(), running_corrects.double()/tot))
                    sys.stdout.flush()
                    sys.stdout.write('\r')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                print('\n| Train Epoch #%d\t\t\tLoss %.4f\tAcc %.4f'
                      % (epoch+1, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid':
                print('| Valid Epoch #%d\t\t\tLoss %.4f\tAcc %.4f'
                      % (epoch+1, epoch_loss, epoch_acc))
                if epoch_acc > best_acc:#and epoch > 80:
                    print('| Saving Best Model...\t\t\t\t\tAcc %.4f' % (epoch_acc))
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, model_filename)

            loss_acc.get_epoch_value(epoch_loss, epoch_acc.cpu().numpy(), lr, phase)
        time_elapsed = time.time() - since
        print('| Epoch Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    loss_acc.save_log(log_filename)
    print('| Best Valid Acc: {:4f}'.format(best_acc))
    print('| Best Model Saved in %s' % model_filename)
    print('| Log Saved in %s' % log_filename)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


print('Loading Model...')
print('-' * 80)
model = get_parallel_model(dataset_class_num, MULTI_GPU, GPU_LIST, DEVICE)

optimizer, criterion, scheduler = init_model(model, LR, LR_STEP, MOMENTUM, WEIGHT_DECAY)

if MODE == 'TRAIN':
    if FINETUNE:
        print('| Load Pretrained Model: %s' % (MODEL_NAME))
    else:
        print('| Load Empty Model: %s' % (MODEL_NAME))
    print()
    best_model = train_model(model, criterion, optimizer, scheduler, EPOCHS)
elif MODE == 'TEST':
    print('| Load Saved Model: %s' % TEST_MODEL_PATH)
    print()
    state_dict = torch.load(TEST_MODEL_PATH)
    model.load_state_dict(state_dict)
    test_model(model, dataloaders['test'], criterion)
