import sys
import os
import os.path as osp
import shutil
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms

sys.path.append('../../')
from text_proc.utils.io import *
# from common.utils import SingleMetrics
from common.utils import BatchMetrics, get_parallel_model
from common.params import *


image_type = ['.jpg']

test_images = '/u2t/wangqiushi/datasets/SNTO_test/snto_test_face_20191101/'
test_labels = '/mnt/datasets02/wangqiushi/mxnet/datasets/hx_xr_xsw_test/label.txt'

test_images = '/u2t/wangqiushi/datasets/CASIA-FASD/shot_images/scale_3.0/test_release/15/4/' 




test_model = '/home/wangqiushi/pycode/pytorch/projects/saved_models/resnet50-CASIA_p1-20191101124819.pth'


test_result_file = './test_result.txt'

classes = [0, 1]
# map_dict = {0:0, 1:1, 2:2, 3:3, 4:3, 5:4, 6:5, 7:6}

# map_dict_inv = {'jmqz':0, 'ky':1, 'ml':2, 'normal':3, 'wsxwy':5, 'xr':6, 'xsw':7}

assert torch.cuda.is_available()
DEVICE = torch.device('cuda:0')


def load_image(image_path):
    normalize = transforms.Normalize(TRAIN_RGB_MEAN,
                                     TRAIN_RGB_SD)
    transform = transforms.Compose([
                    transforms.Resize(RESIZE_SIZE),
                    transforms.CenterCrop(CROP_SIZE),
                    transforms.ToTensor(),
                    normalize])

    img = Image.open(image_path)
    if img is not None:
        img = transform(img)
        img = img.unsqueeze(0)
    return img


def get_model(model_path):
    model = get_parallel_model(len(classes), MULTI_GPU, GPU_LIST, DEVICE)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model



#images, labels = get_image_label_list(test_labels)
image_list = get_legal_root_list(test_images, image_type)

test_size = len(image_list)


def predict_images():
    model = get_model(test_model)

    image_test = []
    image_test_label = []
    for t in range(test_size):
        print('| Test %d:' % (t+1))
        image = image_list[t]
        img = load_image(osp.join(test_images, image))
        img = Variable(img.to(DEVICE))

        score = model(img)
        prob = torch.nn.functional.softmax(score, dim=1)

        max_value, index = torch.max(prob, 1)
        max_value = float(max_value.cpu().detach().numpy())
        index = int(index.cpu().detach().numpy())

        print('| Probability=%.4f, Class=%s' % (max_value, index))

        #for i in a:
        #    print('| Probability=%.4f, Class=%s' %(prob[i], map_dict[i]))

        image_test.append(image)
        image_test_label.append(index)

    print(image_test_label.count(0))
    print(image_test_label.count(1))

#    write_image_label_list(image_test, image_test_label, test_result_file, split=' ')



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
        img = get_image(os.path.join(test_images, image))
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
"""

def main():
    predict_images()

if __name__ == '__main__':
    main()
