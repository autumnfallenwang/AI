from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from matplotlib import pyplot as plt
from common.utils import TrainLossAcc

project_dir = '/home/wangqiushi/pycode/SuperComputer/pytorch/projects/digestive8/'
log_path = project_dir + 'logs/resnet18-test-20180910062504.txt'
plot_dir = project_dir

loss_acc = TrainLossAcc()
loss_acc.read_log(log_path)

def save_plot(loss_acc, plot_dir):
    epochs = len(loss_acc.train_loss)
    epoch_list = list(range(1, epochs+1))

    f1 = plt.figure()
    plt.title('Train/Valid Loss vs. Epoch')
    plt.plot(epoch_list, loss_acc.train_loss, color='red', label='Train Loss')
    plt.plot(epoch_list, loss_acc.valid_loss, color='blue', label='Valid Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(plot_dir, 'Loss_vs_Epoch.png'))
    plt.close(f1)
    
    f2 = plt.figure()
    plt.title('Train/Valid Accuracy vs. Epoch')
    plt.plot(epoch_list, loss_acc.train_acc, color='red', label='Train Accuracy')
    plt.plot(epoch_list, loss_acc.valid_acc, color='blue', label='Valid Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(plot_dir, 'Acc_vs_Epoch.png'))
    plt.close(f2)

    f3 = plt.figure()
    plt.title('Learning Rate vs. Epoch')
    plt.plot(epoch_list, loss_acc.lr, color='red', label='Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(os.path.join(plot_dir, 'LR_vs_Epoch.png'))
    plt.close(f3)

save_plot(loss_acc, plot_dir)