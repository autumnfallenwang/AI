
import os, sys, glob, argparse, datetime

import cv2 
import numpy as np
import mxnet as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../RetinaFace'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../deploy'))

import utils
import face_preprocess
from retinaface import RetinaFace
from mtcnn_detector import MtcnnDetector

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

def get_retinaface_detector(model_str, gpuid):
  _vec = model_str.split(',')
  assert len(_vec)==3
  prefix = _vec[0]
  epoch = int(_vec[1])
  net = _vec[2]
  detector = RetinaFace(prefix, epoch, gpuid, net) 
  return detector

def get_mtcnn_detector(ctx, det, mtcnn_path, thresh = [0.6,0.7,0.8]):
  if det==0:
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=thresh)
  else:
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
  return detector

class SntoFace:
  def __init__(self, args=None):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    self.retinaface_detector = None 
    self.mtcnn_detector = None 

    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')
    if len(args.re_model)>0:
      self.retinaface_detector = get_retinaface_detector(args.re_model, args.gpu)
    if len(args.mt_model)>0:
      self.mtcnn_detector = get_mtcnn_detector(ctx, args.det, args.mt_model)

  def retinaface_detect(self, img, thresh=0.8, scales=[1024, 1980], flip=False):
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    faces, landmarks = self.retinaface_detector.detect(img, thresh, scales=scales, do_flip=flip)
    landmarks = landmarks.transpose((0,2,1)).reshape((landmarks.shape[0], 10))
    return faces, landmarks

  def mtcnn_detect(self, img):
    ret = self.mtcnn_detector.detect_face(img, det_type=self.args.det)
    if ret is None:
      return [], []
    return ret

  def get_aligned_data_from_img(self, img, bbox, points):
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T

    nimg = face_preprocess.preprocess(img, bbox, points, image_size=self.args.image_size)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned 

  def get_aligned_data_from_image_files(self, image_files):
    crops = [] 
    detected_image_files = []
    lost_detected_image_files = []
    for image_path in image_files:
      img = cv2.imread(image_path)
      boundingboxes = []
      points = [] 
      if len(self.args.mt_model)>0:
        boundingboxes, points = self.mtcnn_detect(img)
      else:
        boundingboxes, points = self.retinaface_detect(img)
      if 0 == len(boundingboxes):
        lost_detected_image_files.append(image_path)
        continue
      detected_image_files.append(image_path)
      aligned = self.get_aligned_data_from_img(img, boundingboxes, points)
      crops.append(aligned)
    return np.array(crops), detected_image_files, lost_detected_image_files

  def get_feature(self, aligned):
    data = mx.nd.array(aligned)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embeddings = self.model.get_outputs()[0].asnumpy()
    return embeddings

  def get_ga(self, aligned):
    data = mx.nd.array(aligned)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age 
