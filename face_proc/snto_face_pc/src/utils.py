from __future__ import print_function
import os, sys, copy 
import cv2
import numpy as np

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 

def list_files(image_dir):
    image_files = []
    for cur_dir, sub_dirs, files in os.walk(image_dir):
        for f in files:
            image_files.append(os.path.join(cur_dir, f))
    assert len(image_files) > 0, print("no picture in %s"%image_dir)
    return sorted(image_files)

def draw_bbox(img, bounddingbox, save_path=None, length=2, color = (255,0,0)):
    for i in range(bounddingbox.shape[0]):
        print('score', bounddingbox[i][4])
        box = bounddingbox[i].astype(np.int)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, length)
    if save_path:
        cv2.imwrite(save_path, img)
    return img

def draw_points(img, points, save_path=None, radius=1, length=2, color=(0,0,255)):
    for i in range(points.shape[0]):
        point = points[i].astype(np.int)
        n = int(point.shape[0]/2)
        for j in range(n):
            cv2.circle(img, (point[j], point[j+n]), radius, color, length)
    if save_path:
        cv2.imwrite(save_path, img)
    return img

def cosDist(a, b):
    if len(a.shape) != len(b.shape):
        raise Exception('Input shapes must be the same')
    if not np.equal(a.shape, b.shape):
        raise Exception('Input shapes must be the same')

    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

def euclidDist(a, b):
    if len(a.shape) != len(b.shape):
        raise Exception('Input shapes must be the same')
    if not np.equal(a.shape, b.shape):
        raise Exception('Input shapes must be the same')
    #return np.sqrt(np.sum((a-b)**2))
    return np.linalg.norm(a-b, ord=np.inf)

def getSimilarityMatrix(embs1, embs2, threshold, mode=0): 
    n1 = embs1.shape[0]
    n2 = embs2.shape[0]
    dist = np.zeros((n1, n2))
    sim = np.zeros((n1, n2), dtype=int)
    for i in range(n1):
        for j in range(n2):
            if(0 == mode): 
                dist[i, j] = cosDist(embs1[i], embs2[j])
            elif(1 == mode): 
                dist[i, j] = -1*euclidDist(embs1[i], embs2[j]) 
    sim[np.where(dist>threshold)] = 1
    return dist, sim

def printImages(ref_detected, detected):
    n1 = min(len(ref_detected), len(detected))
    n2 = max(len(ref_detected), len(detected))

    max_len_ref = 0
    for f in ref_detected:
        max_len_ref = max(max_len_ref, len(f))

    max_len = 0
    for f in detected:
        max_len = max(max_len, len(f))

    if n1 == len(detected):
        print('Ref Images'.ljust(max_len_ref+3)+'\tRec Images')
        image_files_list = [ref_detected, detected]
    else:
        print('Rec Images'.ljust(max_len_ref+3)+'\tRef Images')
        image_files_list = [ref_detected, detected]

    for i in range(n1):
        left = '%2d:%s' % (i, image_files_list[0][i])
        print(left.ljust(max_len_ref+3)+'\t%2d:%s'%(i, image_files_list[1][i]))
    for i in range(n1, n2):
        print('%2d:%s' % (i, image_files_list[0][i]))       

def printFloatMatrix(matrix):
    n1 = max(matrix.shape[0], matrix.shape[1])
    n2 = min(matrix.shape[0], matrix.shape[1])
    if n1 == matrix.shape[1]:
        matrix = matrix.T
    for i in range(n2):
        print('\t  %2d  ' % i, end='')
    print('')

    for i in range(n1):
        print('%2d' % i, end='')
        for j in range(n2):
            print('\t%1.4f' % matrix[i, j], end='')
        print('')

def printIntMatrix(matrix):
    n1 = max(matrix.shape[0], matrix.shape[1])
    n2 = min(matrix.shape[0], matrix.shape[1])
    if n1 == matrix.shape[1]:
        matrix = matrix.T 
    print('   ', end=' ')
    for i in range(n2):
        print('%3d'%i, end=' ')
    print(' ')
    for i in range(n1):
        print('%3d'%i, end=' ')
        for j in range(n2):
            print('%3d'%matrix[i, j], end=' ')
        print('')

def statistics(sim, ground_truth, ref_image_files, image_files):
    ref_image_files = np.array(ref_image_files)
    image_files = np.array(image_files)
    lost_rec = (sim & ground_truth) ^ ground_truth
    wrong_rec = (sim | ground_truth) ^ ground_truth 

    n_lost = np.sum(lost_rec)
    n_wrong = np.sum(wrong_rec)

    lost_inds = np.where(np.sum(lost_rec, axis=0)>0)[0]
    wrong_inds = np.where(np.sum(wrong_rec, axis=0)>0)[0] 

    wrong_matches = [] 
    for i in wrong_inds:
        ref_wrong_matches = ref_image_files[np.where(wrong_rec[:, i]==1)]
        wrong_matches.append([image_files[i], ref_wrong_matches])

    return n_lost, n_wrong, image_files[lost_inds], wrong_matches 
    
