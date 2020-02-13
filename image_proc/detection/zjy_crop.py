"""author:zhoujunyan;date:20181221;function:裁剪图片的黑边"""

import cv2
import os
import datetime
import uuid
import numpy as np

def change_size(read_file,file_name):
    image=cv2.imread(read_file,1) #读取图片 image_name应该是变量

    origh, origw, origc = image.shape

    if origw < 100 or origh < 100:
        return

    b=cv2.threshold(image,40,255,cv2.THRESH_BINARY)          #调整裁剪效果
    binary_image=b[1]               #二值图--具有三通道

    #cv2.imwrite(save_path+str(uuid.uuid1())+".jpg",binary_image)

    binary_image=cv2.cvtColor(binary_image,cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)       #改为单通道

    #cv2.imwrite(save_path+str(uuid.uuid1())+".jpg",binary_image)

    x=binary_image.shape[0]
    print("高度x=",x)
    y=binary_image.shape[1]
    print("宽度y=",y)

    paddedimg = binary_image

    h,w = paddedimg.shape
    print('h',h,'w',w)
    kernelsize = max(h,w)//100
    #print 'kernelsize',kernelsize
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelsize,kernelsize))
    #print 'kernel',kernel
    paddedimgerode = cv2.erode(paddedimg, kernel)

    #cv2.imwrite(save_path+str(uuid.uuid1())+".jpg",paddedimgerode)

    r = paddedimgerode

    thresh = 50

    xavgr = np.mean(r, axis=0)#1*n
    yavgr = np.mean(r, axis=1)#m*1
    

    xavgr -= thresh
    yavgr -= thresh
    
    
    xavgrs = np.sign(xavgr)
    yavgrs = np.sign(yavgr)
    

    
    xavgrsd = np.diff(xavgrs)
    yavgrsd = np.diff(yavgrs)
    

    
    zeroxrt = np.where(xavgrsd)#给出数组中值不为零的下标
    zeroyrt = np.where(yavgrsd)#同上
    
    #print 'zeroxrt',zeroxrt,'zeroyrt',zeroyrt
    
    zeroxr = zeroxrt[0]
    zeroyr = zeroyrt[0]

    #print 'zeroxr',zeroxr,'zeroyr',zeroyr
    #print 'len(zeroxr)',len(zeroxr),'len(zeroyr)',len(zeroyr)
    iscrop = 1;
    if len(zeroxr) < 2 and len(zeroyr) < 2:
        x1r = 0
        x2r = origw
        y1r = 0
        y2r = origh
        iscrop = 0;
    elif len(zeroxr) < 2 and len(zeroyr) >= 2:
        x1r = 0
        x2r = origw
        y1r = zeroyr[0]
        y2r = zeroyr[-1]
    elif len(zeroyr) < 2 and len(zeroxr) >= 2:
        y1r = 0
        y2r = origh
        x1r = zeroxr[0]
        x2r = zeroxr[-1]
    else:
        x1r = zeroxr[0]
        x2r = zeroxr[-1]
        y1r = zeroyr[0]
        y2r = zeroyr[-1]

        #scale = 1
        #x1r*=scale
        #x2r*=scale
        #y1r*=scale
        #y2r*=scale
    if iscrop == 1:
        box = [int(x1r), int(y1r), int(x2r), int(y2r)]
        pre1_picture=image[box[1]:box[3], box[0]:box[2]]
        cv2.imwrite(save_path+file_name,pre1_picture)
        if save_crop_box:
            save_str = file_name+','+' '.join(list(map(str, box)))
            with open(crop_box_file, 'a+') as f:
                f.write('%s\n' % save_str)

    else:
        cv2.imwrite(notcrop_path+file_name,image)
    #return pre1_picture                                             #返回图片数据

source_path="/data02/wangqiushi/datasets/DR4/DR2_40_50/new/xr/" #图片来源路径
#source_path="/root/Desktop/testedge/"  
save_path="/data02/wangqiushi/datasets/DR4/DR2_40_50/new/xr_crop/" #图片修改后的保存路径
notcrop_path="/data02/wangqiushi/datasets/DR4/DR2_40_50/new/not_crop/"

save_crop_box = True
crop_box_file = '/data02/wangqiushi/datasets/DR4/DR2_40_50/new/crop_box.txt'

if not os.path.exists(save_path):
    os.mkdir(save_path)

file_names=os.listdir(source_path)

starttime=datetime.datetime.now()
for i in range(len(file_names)):
    print(file_names[i])
    change_size(source_path + file_names[i],file_names[i])        #得到文件名
    #x=change_size(source_path + file_names[i])        #得到文件名
    #cv2.imwrite(save_path+"new_"+file_names[i],x)
    #print("裁剪：",file_names[i])
    #print("裁剪数量:",i)
    #while(i==2600):
        #break


print("裁剪完毕")
endtime = datetime.datetime.now()#记录结束时间
endtime = (endtime-starttime).seconds
print("裁剪总用时",endtime)

