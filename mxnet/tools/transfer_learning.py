from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, time, shutil
from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
# from gluoncv.model_zoo import get_model
from common.utils import get_model
from common.params import *
from common.utils import ImageLabelDataset


classes = 5

epochs = 100
lr = 0.01
per_device_batch_size = 64
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, 40, 50, 60, 70, np.inf]

num_gpus = 1
num_workers = 16
# ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
ctx = [mx.gpu(0)]
#ctx = [mx.gpu(0),  mx.gpu(1), mx.gpu(2), mx.gpu(3)]
batch_size = per_device_batch_size * max(num_gpus, 1)


jitter_param = 0.4
lighting_param = 0.1
TRAIN_RGB_MEAN = [0.5835075779815229, 0.34801369344856836, 0.28106787981796433] # digestive8
TRAIN_RGB_SD = [0.11965080359974416, 0.08724743240200583, 0.07926250478615704]   # digestive8

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_RGB_MEAN, TRAIN_RGB_SD)
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_RGB_MEAN, TRAIN_RGB_SD)
])


image_root = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/train/'
train_label = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/train.txt'
val_label = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/valid.txt'
test_label = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/labels/test.txt'

train_data = gluon.data.DataLoader(
    ImageLabelDataset(image_root, train_label, shuffle=True).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers = num_workers)

val_data = gluon.data.DataLoader(
    ImageLabelDataset(image_root, val_label, shuffle=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = 8)

test_data = gluon.data.DataLoader(
    ImageLabelDataset(image_root, test_label, shuffle=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = 8)


#model_name = 'ResNet50_v2'
#finetune_net = get_model(model_name, pretrained=True)
finetune_net = get_model(class_num=classes, ctx=ctx)

"""
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()
"""

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()


def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()


lr_counter = 0
num_batch = len(train_data)

for epoch in range(epochs):
    print('| epoch: %d' % epoch)
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data):
        if i % 10 == 0:
            print('| batch: [%d/%d]' % (i, num_batch))
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))

_, test_acc = test(finetune_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))
