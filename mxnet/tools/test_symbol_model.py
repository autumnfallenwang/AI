import mxnet as mx
#import mxnet.contrib.onnx as onnx_mxnet
import numpy as np
import os, sys
import shutil
from collections import namedtuple
from common.utils import SingleMetrics
sys.path.append('../../')
from text_proc.utils.io import *


mode = 'label'
#mode = 'folder'

is_save_wrong = False
save_wrong_root = '/u2t/wangqiushi/datasets/SNTO_test/save_wrong_scale4/'


image_type = ['.jpg']

#test_images = '/'
#test_labels = '/u2t/wangqiushi/datasets/CASIA-FASD/labels/scale_4.0/train_p1_valid.txt'

test_images = '/u2t/wangqiushi/datasets/SNTO_test/snto_faces_test_20191104/images/'
test_labels = '/u2t/wangqiushi/datasets/SNTO_test/snto_faces_test_20191104/labels.txt'

#test_images = '/u2t/wangqiushi/datasets/SNTO_test/snto_test_face_20191101/'

test_model = '/home/wangqiushi/pycode/mxnet/projects/export_models/resnet50_v1d-scale_4-20191104152604'

classes = [0, 1]
#classes = [0, 1, 2, 3, 4, 5, 6, 7]

#map_dict = {0:'jmqz', 1:'ky', 2:'ml', 3:'normal', 4:'normal', 5:'wsxwy', 6:'xr', 7:'xsw'}
map_dict = {0:0, 1:1}
#map_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}

map_dict_inv = {'jmqz':0, 'ky':1, 'ml':2, 'normal':3, 'wsxwy':5, 'xr':6, 'xsw':7}

ctx = mx.gpu(0)


sym, arg_params, aux_params = mx.model.load_checkpoint(test_model, 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)

"""
sym, arg_params, aux_params = onnx_mxnet.import_model('resnet50-dg8_nomean_pth040-20180829025439.onnx')
mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['0'], label_names=None)
mod.bind(for_training=False, data_shapes=[('0', (1,3,224,224))])
mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
"""


def get_image(image):
    # download and show the image
    img = mx.image.imread(image)
    if img is None:
        return None
    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    
    """
    # pytorch onnx model
    img = mx.nd.array(img, dtype=np.float32)
    img = img / 255
    """
    """
    # changsha model
    img = mx.nd.array(img, dtype=np.float32)
    img = img / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    """

    img = img.expand_dims(axis=0) # batchify
    return img


if mode == 'label':
    images, labels = get_image_label_list(test_labels, False)
elif mode == 'folder':
    images = get_legal_root_list(test_images, image_type)

test_size = len(images)

Batch = namedtuple('Batch', ['data'])


def predict_images():
    image_test = []
    image_test_label = []

    for t in range(test_size):
        print('| Test %d:' % (t+1))
        image = images[t]
        img = get_image(osp.join(test_images, image))
        img = img.copyto(ctx)

        mod.forward(Batch([img]))
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        for i in a:
            print('| Probability=%.4f, Class=%s' %(prob[i], map_dict[i]))

        pred = map_dict[a[0]]
        prob0 = float(prob[a[0]])
        image_test.append(image)
        image_test_label.append(pred)

        if is_save_wrong:
            save_label_root = osp.join(save_wrong_root, str(pred))
            os.makedirs(save_label_root, exist_ok=True)
            img_filename = osp.splitext(osp.split(image)[1])[0]+'_'+str(pred)+'_'+str(prob0)+'.jpg'
            save_img_path = osp.join(save_label_root, img_filename)
            print('| Copy to %s' % save_img_path)
            shutil.copy(osp.join(test_images, image), save_img_path)

"""
    with open(save_path, 'w+') as f:
        for i in range(len(image_test)):
            f.write('%s %s\n' % (image_test[i], image_test_label[i]))
"""

def predict():
    correct_all = 0
    image_correct = []
    image_prob = []
    image_label = []

    metrics = SingleMetrics(classes)

    for t in range(test_size):
        print('| Test %d:' % (t+1))
        image = images[t]
        label = labels[t]
        img = get_image(osp.join(test_images, image))
        img = img.copyto(ctx)

        mod.forward(Batch([img]))
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        for i in a:
            print('| Probability=%.4f, Class=%s' %(prob[i], str(i)))
        print('| Label = %s' % label)

        pred = map_dict[a[0]]
        label = map_dict[label]
#        pred = a[0]
#        label = label

        metrics.get_single_cf_matrix(pred, label)

        if pred == label:
            print('| Y')
            correct_all += 1
            image_correct.append(image)
            image_prob.append(prob[a[0]])
            image_label.append(label)
        else:
            print('| N')
        acc = float(correct_all)/(t+1)
        print('| Accuracy = %.4f\n' % acc)

    metrics.get_metrics()
    metrics.print_metrics()
    metrics.print_cf_matrix()
    
    accuracy_all = float(correct_all)/(test_size)
    print('| Test Model: %s' % test_model)
    print('| Accuracy_all = %.4f' % accuracy_all)
    
#    for i in range(8):
#        print('Label %d = %d' % (i, image_label.count(i)))


def main():
    if mode == 'label':
        predict()
    elif mode == 'folder':
        predict_images()


if __name__ == '__main__':
    main()
