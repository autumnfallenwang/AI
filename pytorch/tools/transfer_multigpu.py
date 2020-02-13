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
from tqdm import tqdm
from common.utils_old import get_cuda_version, get_cudnn_version, get_gpu_name
from common.utils import ClsDataset, get_parallel_model
from common.metrics import ClsMetrics
from common.log import ClsLog
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
    x : ClsDataset(image_root=IMAGE_ROOT[x],
                   label_path=LABEL_PATH[x],
                   shuffle=True,
                   transform=transform[x])
    for x in ['train', 'valid', 'test']
}
for x in ['train', 'valid', 'test']:
    print('| '+x+': '+LABEL_PATH[x])
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


def test_model(model, dataloader, criterion, metrics):
    model.eval()

    dataset_size = len(dataloader.dataset)

    running_loss = 0.0

    pred_list = np.array([], dtype=int)
    label_list = np.array([], dtype=int)

    iters = (dataset_size // BATCHSIZE) + 1
    pbar = tqdm(total=iters)
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
            # Save loss
            running_loss += loss.item() * inputs.size(0)
            # Save preds
            pred_list = np.append(pred_list, preds.cpu().numpy())
            label_list = np.append(label_list, labels.cpu().numpy())

            pbar.set_description('| Testing')
            pbar.update(1)

    pbar.close()
    # Final loss
    epoch_loss = running_loss / dataset_size
    metrics.set('test_loss', epoch_loss)
    metrics.calculate(label_list, pred_list)


def train_model(model, criterion, optimizer, scheduler, metrics, log, epochs):
    print('Training Model...')
    best_model_wts = copy.deepcopy(model.state_dict())

    os.makedirs(SAVED_MODELS, exist_ok=True)
    os.makedirs(SAVED_LOGS, exist_ok=True)

    time_filename = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model_filename = os.path.join(SAVED_MODELS, SAVED_PREFIX+time_filename+'.pth')
    log_filename = os.path.join(SAVED_LOGS, SAVED_PREFIX+time_filename+'.csv')

    for epoch in range(epochs):
        since = time.time()
        print('-' * 80)

        log_item_dict = {'epoch':[], 'time':[], 'lr':[], 'train_loss':[], 'train_acc':[]}

        lr = optimizer.param_groups[0]['lr']
        print('| Epoch [%2d/%2d], LR=%f' %(epoch+1, epochs, lr))
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        
        iters = (dataset_sizes['train'] // BATCHSIZE) + 1
        pbar = tqdm(total=iters)
        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            pbar.set_description('| Training')
            pbar.update(1)

        pbar.close()
        scheduler.step()

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        print('| Train Loss: %.4f, Accurary: %.4f' % (epoch_loss, epoch_acc))

        test_model(model, dataloaders['valid'], criterion, metrics)
        print('| Valid Metrics:')
        metrics.print()

        # deep copy the model
        sign = '>' if SAVE_BEST[1] == 'max' else '<'
        save_metrics = metrics.get(SAVE_BEST[0])
        if epoch == 0:
            best_metrics = save_metrics

        elif eval(str(save_metrics) + sign + str(best_metrics)): #and epoch > 80:
            print('| Saving Best Model...  '+SAVE_BEST[0]+': %.4f' % (save_metrics))
            best_metrics = save_metrics
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, model_filename)

        time_elapsed = time.time() - since
        time_string = '%.0fm%.0fs' % (time_elapsed // 60, time_elapsed % 60)
        print('| Epoch Complete in %s' % time_string)
        print()

        # log item
        log_item_dict['epoch'] = [epoch]
        log_item_dict['time'] = [time_string]
        log_item_dict['lr'] = [lr]
        log_item_dict['train_loss'] = [epoch_loss]
        log_item_dict['train_acc'] = [np.float64(epoch_acc.cpu().numpy())]
        for log_index in log.columns:
            if log_index not in log_item_dict.keys():
                if log_index in metrics.metrics_result.keys():
                    log_item_dict[log_index] = [metrics.get(log_index)]
        log.save(log_item_dict)
        log.write(log_filename)

    print('| Best '+SAVE_BEST[0]+': %.4f' % (best_metrics))
    print('| Best Model Saved in %s' % model_filename)
    print('| Log Saved in %s' % log_filename)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


print('Loading Model...')
print('-' * 80)
model = get_parallel_model(dataset_class_num, MULTI_GPU, GPU_LIST, DEVICE)

optimizer, criterion, scheduler = init_model(model, LR, LR_STEP, MOMENTUM, WEIGHT_DECAY)

metrics = ClsMetrics(METRICS)

log = ClsLog(LOG)

if MODE == 'TRAIN':
    if FINETUNE:
        print('| Load Pretrained Model: %s' % (MODEL_NAME))
    else:
        print('| Load Empty Model: %s' % (MODEL_NAME))
    print()
    best_model = train_model(model, criterion, optimizer, scheduler, metrics, log, EPOCHS)

elif MODE == 'TEST':
    print('| Load Saved Model: %s' % TEST_MODEL_PATH)
    print()
    state_dict = torch.load(TEST_MODEL_PATH)
    model.load_state_dict(state_dict)
    test_model(model, dataloaders['test'], criterion, metrics)
    metrics.print()
