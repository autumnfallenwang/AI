import os, sys, argparse

import utils
from sntoFace import SntoFace 

import cv2
import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='face model test')
  parser.add_argument('--image-size', default='112,112', help='')
  parser.add_argument('--model', default='', help='path to load model.')
  parser.add_argument('--ga-model', default='', help='path to load model.')
  parser.add_argument('--re-model', default='', help='path to load model.')
  parser.add_argument('--mt-model', default='/u2t/share/insightface/model/FaceDetection/mtcnn-model/model_mxnet', help='path to load model.')
  parser.add_argument('--gpu', default=0, type=int, help='gpu id')
  parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
  args = parser.parse_args()

  snto_model = SntoFace(args)
  
  img_path='/u2t/wangqiushi/datasets/NUAA/raw/ClientRaw/0002/0002_01_00_01_100.jpg'
  img = cv2.imread(img_path)
  bbox, points = snto_model.mtcnn_detect(img)
  print(bbox)
  print(points)
