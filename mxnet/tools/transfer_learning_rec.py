from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil, time
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import autograd as ag
from gluoncv.utils import LRScheduler
from common.utils import get_model
from common.params import *

classes = 5

epochs = 200

per_device_batch_size = 64
momentum = 0.9
wd = 0.0001

lr_mode = 'cosine'
lr = 0.4
lr_decay = 0.1
lr_decay_period = 0
lr_decay_epoch = '40, 60'
# warmup_epochs = 5

#lr_steps = [10, 20, 30, 40, 50, 60, 70, 80, np.inf]

#lr_factor = 0.1
#lr_steps = [50, 80, np.inf]

num_gpus = 8
num_workers = 16

#ctx = [mx.gpu(0)]
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
#ctx = [mx.gpu(0),  mx.gpu(1), mx.gpu(2), mx.gpu(3)]
#ctx = [mx.gpu(4),  mx.gpu(5), mx.gpu(6), mx.gpu(7)]
batch_size = per_device_batch_size * max(num_gpus, 1)

rec_root = '/raid/data/kaggle/diabetic_retinopathy/mxnet_rec/rec/'
num_training_samples = 28099

jitter_param = 0.4
lighting_param = 0.1
# mean_rgb = [123.68, 116.779, 103.939]
# std_rgb = [58.393, 57.12, 57.375]
mean_rgb = [148.79, 88.74, 71.67]
std_rgb = [30.51, 22.25, 20.21]

data_shape = (3, 224, 224)

train_data = mx.io.ImageRecordIter(
    path_imgrec         = rec_root+'train.rec',
    path_imgidx         = rec_root+'train.idx',
    preprocess_threads  = num_workers,
    shuffle             = True,
    batch_size          = batch_size,

    data_shape          = data_shape,
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
    rand_mirror         = True,
    random_resized_crop = True,
    max_aspect_ratio    = 4. / 3.,
    min_aspect_ratio    = 3. / 4.,
    max_random_area     = 1,
    min_random_area     = 0.08,
    brightness          = jitter_param,
    saturation          = jitter_param,
    contrast            = jitter_param,
    pca_noise           = lighting_param,
    )

val_data = mx.io.ImageRecordIter(
    path_imgrec         = rec_root+'valid.rec',
    path_imgidx         = rec_root+'valid.idx',
    preprocess_threads  = 4,
    shuffle             = False,
    batch_size          = batch_size,

    resize              = data_shape[1],
    data_shape          = data_shape,
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
#    brightness          = jitter_param,
#    saturation          = jitter_param,
#    contrast            = jitter_param,
#    pca_noise           = lighting_param,
    )

test_data = mx.io.ImageRecordIter(
    path_imgrec         = rec_root+'test.rec',
    path_imgidx         = rec_root+'test.idx',
    preprocess_threads  = 4,
    shuffle             = False,
    batch_size          = batch_size,

    resize              = data_shape[1],
    data_shape          = data_shape,
    mean_r              = mean_rgb[0],
    mean_g              = mean_rgb[1],
    mean_b              = mean_rgb[2],
    std_r               = std_rgb[0],
    std_g               = std_rgb[1],
    std_b               = std_rgb[2],
#    brightness          = jitter_param,
#    saturation          = jitter_param,
#    contrast            = jitter_param,
#    pca_noise           = lighting_param,
    )


finetune_net = get_model(class_num=classes, ctx=ctx)

if lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')]
num_batches = num_training_samples // batch_size
lr_scheduler = LRScheduler(mode=lr_mode, baselr=lr,
                           niters=num_batches, nepochs=epochs,
                           step_epoch=lr_decay_epoch, step_factor=lr_decay, power=2)

trainer = gluon.Trainer(finetune_net.collect_params(), 'nag', {
                        'lr_scheduler': lr_scheduler, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=True)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=True)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

"""
test_num = 1
vacc = []
tacc = []
for i in range(test_num):
    print('| [%d/%d]' % (i, test_num))
    
    val_data.reset()
    _, val_acc = test(finetune_net, val_data, ctx)
    print('| [Finished] Val-acc: %.4f' % (val_acc))
    vacc.append(val_acc)

    test_data.reset()
    _, test_acc = test(finetune_net, test_data, ctx)
    print('| [Finished] Test-acc: %.4f' % (test_acc))
    tacc.append(test_acc)

print('| Average VACC = %.4f' % (sum(vacc)/len(vacc)))
print('| Average TACC = %.4f' % (sum(tacc)/len(tacc)))
exit(0)
"""

#lr_counter = 0

best_val_acc = 0.0

os.makedirs(SAVED_MODELS, exist_ok=True)
os.makedirs(SAVED_LOGS, exist_ok=True)
time_filename = time.strftime("%Y%m%d%H%M%S", time.localtime())
model_filename = os.path.join(SAVED_MODELS, SAVED_PREFIX+time_filename+'.params')

for epoch in range(epochs):
    print('| epoch: %d' % epoch)
#    if epoch == lr_steps[lr_counter]:
#        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
#        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()
    train_data.reset()
    val_data.reset()
    for i, batch in enumerate(train_data):
        if i % 50 == 0:
            print('| batch: [%d/%d]' % (i, num_batches))
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=True)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=True)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        lr_scheduler.update(i, epoch)
        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batches

    _, val_acc = test(finetune_net, val_data, ctx)

    print('| [Epoch %d] lr: %.4g, Train-acc: %.4f, loss: %.4f | Val-acc: %.4f | time: %.1f' %
             (epoch, lr_scheduler.learning_rate, train_acc, train_loss, val_acc, time.time() - tic))
    
    if val_acc > best_val_acc:
        print('| Saving Best Model...\tVal-acc: %.4f' % (val_acc))
        best_val_acc = val_acc
        finetune_net.save_parameters(model_filename)

_, test_acc = test(finetune_net, test_data, ctx)
print('| [Finished] Test-acc: %.4f' % (test_acc))

print('| Best Valid Acc: %.4f' % best_val_acc)
print('| Best Model Saved in %s' % model_filename)
