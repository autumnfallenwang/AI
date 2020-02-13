import os
import shutil
import time
import mxnet as mx
import numpy as np
import gluoncv as gcv
import gluoncv.data as gdata
import gluoncv.utils as gutils
from common.utils import SingleMetrics

ctx = mx.gpu(2)

#test_labels = '/mnt/datasets02/wangqiushi/mxnet/datasets/video_test/images2.txt'
#test_images = '/mnt/datasets02/wangqiushi/mxnet/datasets/video_test/'

#test_labels = '/data02/wangqiushi/datasets/digestive8/val.txt'
#test_images = '/data02/wangqiushi/datasets/digestive8/val/'
#map_dict_inv = {'0':0, '1':1, '2':2, '3':3, '4':3, '5':4, '6':5, '7':6}

test_labels = '/data02/wangqiushi/datasets/hx_xr_xsw_test/hx_test/all20190305/images_RB/labels.txt'
test_images = '/'
#map_dict_inv = {'jmqz':2, 'ky':2, 'ml':2, 'xr':0, 'xsw':1, 'normal':2}
map_dict_inv = {'jmqz':0, 'ky':1, 'ml':2, 'xr':3, 'xsw':4, 'normal':5}
#label_map = {0:'jmqz', 1:'ky', 2:'ml', 3:'xr', 4:'xsw', 5:'normal'}
#label_map = {0:'xr', 1:'xsw', 2:'normal'}

#test_labels = '/data02/wangqiushi/datasets/hx_xr_xsw_test/remove_edge/label.txt'
#test_images = '/data02/wangqiushi/datasets/hx_xr_xsw_test/remove_edge/pick/'

#test_images = '/mnt/datasets02/wangqiushi/mxnet/datasets/hx_xr_xsw_test/pick'
#test_labels = '/mnt/datasets02/wangqiushi/mxnet/datasets/hx_xr_xsw_test/label.txt'
#map_dict_inv = {'jmqz':0, 'ky':1, 'ml':1, 'normal':2, 'wsxwy':3, 'xr':4, 'xsw':5}

#map_dict = {0:'jmqz', 1:'ky', 2:'ml', 3:'normal', 4:'normal', 5:'wsxwy', 6:'xr', 7:'xsw'}
#map_dict = {'10':'jmqz', '20':'ky', '30':'ml', '70':'wsxwy', '40':'xr', '50':'xsw'}
#{0:'jmqz'(0), 1:'ky'(1), 2:'ml'(1), 3:'normal'(2), 4:'normal'(2), 5:'wsxwy'(3), 6:'xr'(4), 7:'xsw'(5)}

classes = [0, 1, 2, 3, 4, 5]
#classes = [0, 1, 2]

model_path = '/mnt/datasets02/wangqiushi/mxnet/projects/DR2_40_50/new/export_models/ssd_512_resnet50_v1_voc_best'

pred_map_dict = {0:3, 1:4, -1:5}
#pred_map_dict = {0:0, 1:1, -1:2}

normal_threshold = {0:0.5, 1:0.3, -1:0.0}

"""
model_path = '/mnt/datasets02/wangqiushi/mxnet/projects/DR4/export_models/ssd_512_resnet50_v1_voc_best'

pred_map_dict = {0:1, 1:2, 2:3, 3:4, -1:5}
normal_threshold = [0.1, 0.1, 0.5, 0.5]
"""

is_save_npz = False
save_npz_root = '/data02/wangqiushi/datasets/hx_xr_xsw_test/hx_test/all20190305/data/ssd_DR2_new/'

is_save_wrong = False
save_wrong_root = '/data02/wangqiushi/datasets/hx_xr_xsw_test/hx_test/all20190305/wrong3/'

model = mx.gluon.nn.SymbolBlock.imports(model_path+'-symbol.json', ['data'], model_path+'-0000.params', ctx=ctx)


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
#            labels.append(int(string[1].rstrip('\n')))
            labels.append(map_dict_inv[string[1].rstrip('\n')])
    return images, labels

def get_image_list(label_file):
    images = []
    with open(label_file, 'r') as read_batch:
        while True:
            lines = read_batch.readline()
            if not lines:
                break
            images.append(lines.rstrip('\n'))
    return images

def save_npz(npz_path, class_IDs, scores):
    ID = class_IDs.asnumpy()
    score = scores.asnumpy()

    np.savez(npz_path, ID=ID, score=score)


images, labels = get_image_label_list(test_labels)
#images = get_image_list(test_labels)

test_size = len(images)

def get_image(image):
    img = mx.image.imread(image)
    if img is None:
        return None
    
    img = gdata.transforms.image.resize_short_within(img, short=512, max_size=1024) # resize short

    img = img.transpose((2, 0, 1)) # Channel first
    img = mx.nd.array(img, dtype=np.float32)
    img = img / 255
    
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c] 

    img = img.expand_dims(axis=0) # batchify
    return img

def thres_label(class_IDs, scores, thres_dict):
    lenth = max(class_IDs.shape)
    ID = class_IDs.reshape(lenth,)
    score = scores.reshape(lenth,)

    thres_map = np.array(list(map(lambda x:thres_dict[x], ID)))
    mask = score > thres_map
    ID = ID[mask]
    score = score[mask]

    pred0 = -1
    prob0 = 0.0
    if len(ID) != 0:
        pred0 = int(ID[0])
        prob0 = float(score[0])

    return pred0, prob0
        

def predict():
    time_sum = 0.0
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

        x = get_image(img)
        
        x = x.copyto(ctx)

        time_s = time.time()
        #mod.forward(Batch([img]))
        class_IDs, scores, bounding_boxs = model(x)
        time_e = time.time() - time_s

        if is_save_npz:
            npz_file = os.path.split(img)[1].replace('.jpg', '.npz')
            npz_path = os.path.join(save_npz_root, npz_file)
            save_npz(npz_path, class_IDs[0], scores[0])

        pred0, prob0 = thres_label(class_IDs[0].asnumpy(), 
                                   scores[0].asnumpy(), 
                                   normal_threshold)

        pred = pred_map_dict[pred0]

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
            if is_save_wrong:
                save_label_root = os.path.join(save_wrong_root, label_map[label])
                os.makedirs(save_label_root, exist_ok=True)
                img_filename = label_map[pred]+'_'+os.path.split(img)[1]
                save_img_path = os.path.join(save_label_root, img_filename)
                print('| Copy to %s' % save_img_path)
                shutil.copy(img, save_img_path)

        acc = float(correct_all)/(t+1)
        print('| Accuracy = %.4f' % acc)

        time_sum += time_e
        avg_spf = time_sum/(t+1)
        print('| Average SPF = %.4f seconds\n' % avg_spf)

    metrics.get_metrics()
    metrics.print_metrics()
    metrics.print_cf_matrix()
    
    accuracy_all = float(correct_all)/(test_size)
    print('| Test Model: %s' % model_path)
    print('| Accuracy_all = %.4f' % accuracy_all)
    print('| Average SPF = %.4f seconds\n' % avg_spf)


def predict_images():
    image_test = []
    image_test_label = []
    image_test_prob = []

    for t in range(test_size):
        print('| Test %d:' % (t+1))
        image = images[t]

        x = get_image(test_images+image)
        x = x.copyto(ctx)

        class_IDs, scores, bounding_boxs = model(x)

        prob0 = float(scores[0][0].asnumpy())

        pred0 = int(class_IDs[0][0].asnumpy())

        if prob0 > normal_threshold:
            pred = pred0
        else:
            pred = -1

        pred = pred_map_dict[pred]
        
        image_test.append(image)
        image_test_label.append(pred)
        image_test_prob.append(prob0)

        print('| Image = %s' % image)
        print('| Pred = %s' % pred)
        print('| Prob = %.4f' % prob0)

#    with open('pred_label.txt', 'w+') as f:
#        for i in range(len(image_test_label)):
#            f.write('%s\n' % image_test_label[i])
    for lb in image_test_label:
        print('%s' % lb)
    for prob in image_test_prob:
        print('%.4f' % prob)


def main():
    predict()

if __name__ == '__main__':
    main()