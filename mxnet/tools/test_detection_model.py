import os
import shutil
import mxnet as mx
import numpy as np
import gluoncv as gcv
import gluoncv.data as gdata
import gluoncv.utils as gutils
from common.utils import SingleMetrics

ctx = mx.gpu(0)

test_labels = '/data02/wangqiushi/datasets/digestive8/test.txt'
test_images = '/data02/wangqiushi/datasets/digestive8/test/'

#map_dict = {0:'jmqz', 1:'ky', 2:'ml', 3:'normal', 4:'normal', 5:'wsxwy', 6:'xr', 7:'xsw'}
#map_dict = {'10':'jmqz', '20':'ky', '30':'ml', '70':'wsxwy', '40':'xr', '50':'xsw'}

label_map_dict = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:4, 7:5}
#{0:'jmqz'(0), 1:'ky'(1), 2:'ml'(1), 3:'normal'(2), 4:'normal'(2), 5:'wsxwy'(3), 6:'xr'(4), 7:'xsw'(5)}
classes = [0, 1, 2, 3, 4, 5]

model_name = 'ssd_512_resnet50_v1_voc'

model_params = '/mnt/datasets02/wangqiushi/mxnet/tools/ssd_512_resnet50_v1_voc_0110_0.6158.params'
model_classes = ('30', '40', '50')

pred_map_dict = {0:1, 1:4, 2:5, 3:2}

model = gcv.model_zoo.get_model(model_name, pretrained_base=True)
model.reset_class(model_classes)

model.load_parameters(model_params)

model.collect_params().reset_ctx(ctx)

def get_image_label_list(label_file):
    images = []
    labels = []
    with open(label_file, 'r') as read_batch:
        while True:
            lines = read_batch.readline()
            if not lines:
                break
            string = lines.split()
            images.append(string[0])
            labels.append(int(string[1].rstrip('\n')))
#            labels.append(map_dict_inv[string[1].rstrip('\n')])
    return images, labels

images, labels = get_image_label_list(test_labels)

test_size = len(images)


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
        img = os.path.join(test_images, image)

        x, _ = gdata.transforms.presets.ssd.load_test(img, short=512)
        
        x = x.copyto(ctx)

        #mod.forward(Batch([img]))
        class_IDs, scores, bounding_boxs = model(x)

        prob0 = float(scores[0][0].asnumpy())

        #prob = np.squeeze(prob)
        #a = np.argsort(prob)[::-1]
        #for i in a:
        #    print('| Probability=%.4f, Class=%s' %(prob[i], str(i)))
        #print('| Label = %s' % label)

        pred0 = int(class_IDs[0][0].asnumpy())

        if prob0 > 0.3:
            pred = pred0
        else:
            pred = 3

        pred = pred_map_dict[pred]
        label = label_map_dict[label]

        print('| Pred = %d' % pred)
        print('| Label = %d' % label)
        print('| Prob = %.4f' % prob0)

        metrics.get_single_cf_matrix(pred, label)

        if pred == label:
            print('| Y')
            correct_all += 1
#            image_correct.append(image)
#            image_prob.append(prob[a[0]])
#            image_label.append(label)
        else:
            print('| N')
        acc = float(correct_all)/(t+1)
        print('| Accuracy = %.4f\n' % acc)

    metrics.get_metrics()
    metrics.print_metrics()
    metrics.print_cf_matrix()
    
    accuracy_all = float(correct_all)/(test_size)
    print('| Test Model: %s' % model_params)
    print('| Accuracy_all = %.4f' % accuracy_all)


def main():
    predict()

if __name__ == '__main__':
    main()