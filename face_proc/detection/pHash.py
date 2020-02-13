import os
import cv2
import numpy as np
import sys


def get_root_list(root):
    files = []
    for root, dirs, file in os.walk(root):  
    #    print(root) #当前目录路径 
    #    print(dirs) #当前路径下所有子目录
    #    print(file)
        files.append(file)
    return files[0]

def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    img = cv2.imread(imgfile, 0) 
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)

        #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32, 32)

    #把二维list变成一维list
    img_list = vis1.flatten().tolist() 

    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])

'''
cv2.imread
flags>0时表示以彩色方式读入图片 
flags=0时表示以灰度图方式读入图片 
flags<0时表示以图片的本来的格式读入图片

interpolation - 插值方法。共有5种：
１）INTER_NEAREST - 最近邻插值法
２）INTER_LINEAR - 双线性插值法（默认）
３）INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。
对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
４）INTER_CUBIC - 基于4x4像素邻域的3次插值法
５）INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值
http://blog.csdn.net/u012005313/article/details/51943442
'''
def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

def similar_rate(image1, image2):
    s1 = pHash(image1)
    s2 = pHash(image2)
    return 1 - hammingDist(s1, s2)*1. / (32*32/4)

def distribute(np_list, x0, x1):
    re_list = np_list[(np_list>x0)*(np_list<=x1)]
    return len(re_list)

root_c = '/data02/wangqiushi/datasets/digestive8/test/'

image_c = os.path.join(root_c, '0_jmqz_100000000.jpg')

root_t = '/data02/wangqiushi/datasets/digestive8/train/'

thres = 0.9

image_ts = get_root_list(root_t)

sr_list = []

for i, image_t in enumerate(image_ts):
    print('%d/%d' % (i+1, len(image_ts)))
    print(image_t)
    sr_list.append(similar_rate(image_c, os.path.join(root_t, image_t)))

sr_list = np.array(sr_list)

for i in range(10):
    x0 = i * 0.1
    x1 = (i+1) * 0.1
    print('%2f < x <= %2f: %d' % (x0, x1, distribute(sr_list, x0, x1)))









