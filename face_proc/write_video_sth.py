import os, sys
from utils.write_video import write_to_video


video_path = './faces/4.avi'
model_path = './mxnet_models/resnet50_v1d-scale_4-20191104152604'
save_path = './test_fake.avi'
scale = 3.0
device = 'cpu'

write_to_video(video_path=video_path,
               model_path=model_path,
               save_path=save_path,
               scale=scale,
               device=device)

