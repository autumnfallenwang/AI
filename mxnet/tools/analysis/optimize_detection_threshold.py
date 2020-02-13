import os
import shutil
import sys
import time
import numpy as np
from scipy.optimize import minimize
sys.path.append('../')
from common.utils import SingleMetrics

test_labels = '/data02/wangqiushi/datasets/hx_xr_xsw_test/hx_test/all20190305/images_RB/labels_hx_test_orig.txt'
test_images = '/'
map_dict_inv = {'jmqz':2, 'ky':2, 'ml':2, 'xr':0, 'xsw':1, 'normal':2}

save_npz_root = '/data02/wangqiushi/datasets/hx_xr_xsw_test/hx_test/all20190305/data/ssd_DR2_new/'

classes = [0, 1, 2]

pred_map_dict = {0:0, 1:1, -1:2}

#normal_threshold = {0:0.5, 1:0.8, -1:0.0}

#mode = 'all'
#mode = 'avg'
mode = 'test'

x0 = [0.6, 0.30]
#x0 = [0.7527, 0.9641]
#x0 = [0.4780, 0.7814]


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
            labels.append(map_dict_inv[string[1].rstrip('\n')])
    return images, labels

def load_npz(npz_path):
    A = np.load(npz_path)
    return A['ID'], A['score']

def thres_list_to_dict(thres_list):
    lenth = len(thres_list)
    thres_dict = {-1:0.0}
    for key in range(lenth):
        thres_dict[key] = thres_list[key]
    return thres_dict

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


images, labels = get_image_label_list(test_labels)
test_size = len(images)

def predict(thres_list):

    correct_all = 0
    metrics = SingleMetrics(classes)

    for t in range(test_size):
        image = images[t]
        label = labels[t]
        img = os.path.join(test_images, image)

        npz_file = os.path.split(img)[1].replace('.jpg', '.npz')
        npz_path = os.path.join(save_npz_root, npz_file)


        class_IDs, scores = load_npz(npz_path)



        pred0, prob0 = thres_label(class_IDs, scores, thres_list_to_dict(thres_list))
        pred = pred_map_dict[pred0]

        metrics.get_single_cf_matrix(pred, label)

        if pred == label:
            correct_all += 1

    accuracy_all = float(correct_all)/(test_size)

    metrics.get_metrics()
    recalls = []
    for cl in metrics.classes:
        TP = metrics.metrics[cl, 0]
        FP = metrics.metrics[cl, 1]
        TN = metrics.metrics[cl, 2]
        FN = metrics.metrics[cl, 3]
        recalls.append(TP/(TP + FN))
    accuracy_avg = sum(recalls)/len(recalls)

    if mode == 'all':
        return (1 - accuracy_all)
    elif mode == 'avg':
        return (1 - accuracy_avg)
    elif mode == 'test':
        metrics.print_metrics()
        metrics.print_cf_matrix()
        print('| Accuracy_all = %.4f' % accuracy_all)
        print('| accuracy_avg = %.4f' % accuracy_avg)


def optim():
    print('| Mode: %s' % mode)
    t0 = time.time()
    predict(x0)
    print('| x0 = %s' % str(x0))
    print('| One func time: %ds\n' % int(time.time()-t0))
   
    def callback(x):
        for i in range(len(x)):
            print('| x%d = %.4f' % (i, x[i]))
        print('| fmin = %.4f\n' % predict(x))

    if mode != 'test':
        res = minimize(predict, x0, 
                       method='Nelder-Mead', 
                       tol=1e-4, 
                       callback=callback, 
                       options={'disp': True})

def main():
    optim()

if __name__ == '__main__':
    main()
