#MODE = 'TRAIN'
MODE = 'TEST'

MULTI_GPU = False
GPU_DEVICE = 0 # single use
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
GPU_COUNT = len(GPU_LIST)

EPOCHS = 60

BATCHSIZE = 32
# LR = 0.0001
# LR_STEP = 10
LR = 0.01
LR_STEP = 20
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3

if MULTI_GPU:
    LR *= GPU_COUNT 
    BATCHSIZE *= GPU_COUNT

RESIZE = True
RESIZE_SIZE = (256, 256) # H * W
CROP_SIZE = (224, 224)

"""
# digesetive8
DATA_PATH = '/workspace/wangqiushi/datasets/digestive8/'
IMAGE_DIR = {
    'train': DATA_PATH + 'raw_256_256',
    'valid': DATA_PATH + 'raw_256_256',
    'test':  DATA_PATH + 'raw_256_256',
   # 'test': '/workspace/wangqiushi/datasets/digestive8/val/',
}
LABEL_FILES = {
    'train': DATA_PATH + 't80/train.txt',
    'valid': DATA_PATH + 't80/valid.txt',
    'test': DATA_PATH + 't80/test.txt',
   # 'test': '/workspace/wangqiushi/datasets/digestive8/val.txt',
}
TRAIN_RGB_MEAN = [0.5835075779815229, 0.34801369344856836, 0.28106787981796433] # digestive8
TRAIN_RGB_SD = [0.11965080359974416, 0.08724743240200583, 0.07926250478615704]   # digestive8
# TRAIN_RGB_MEAN = [0.29840457552695243, 0.20021442443200835, 0.16448449668955578] # d_huaxi_test
# TRAIN_RGB_SD = [0.12407090998832618, 0.10567295523101511, 0.10110819232370424] # d_huaxi_test
"""

# digesetive8
DATA_PATH = '/workspace/wangqiushi/datasets/d8_region_mix/xr_normal/'
IMAGE_DIR = {
    'train': '/',
    'valid': '/',
    'test':  '/',
   # 'test': '/workspace/wangqiushi/datasets/digestive8/val/',
}
LABEL_FILES = {
    'train': DATA_PATH + 't90/train.txt',
    'valid': DATA_PATH + 't90/valid.txt',
    'test': DATA_PATH + 't90/valid.txt',
   # 'test': '/workspace/wangqiushi/datasets/digestive8/val.txt',
}
TRAIN_RGB_MEAN = [0.5835075779815229, 0.34801369344856836, 0.28106787981796433] # digestive8
TRAIN_RGB_SD = [0.11965080359974416, 0.08724743240200583, 0.07926250478615704]   # digestive8
# TRAIN_RGB_MEAN = [0.29840457552695243, 0.20021442443200835, 0.16448449668955578] # d_huaxi_test
# TRAIN_RGB_SD = [0.12407090998832618, 0.10567295523101511, 0.10110819232370424] # d_huaxi_test

"""
# DogsvsCats
DATA_PATH = '/workspace/wangqiushi/datasets/DogsvsCats/'
IMAGE_DIR = {
    'train': DATA_PATH + 'train',
    'valid': DATA_PATH + 'train',
    'test': DATA_PATH + 'train',
}
LABEL_FILES = {
    'train': DATA_PATH + 'train.txt',
    'valid': DATA_PATH + 'valid.txt',
    'test': DATA_PATH + 'test.txt',
}
"""
"""
# imagenet2012
DATA_PATH = '/workspace/wangqiushi/datasets/imagenet2012/raw_256_256/'
IMAGE_DIR = {
    'train': DATA_PATH + 'train/',
    'valid': DATA_PATH + 'val/',
    'test': DATA_PATH + 'val/',
}
LABEL_FILES = {
    'train': DATA_PATH + 'train.txt',
    'valid': DATA_PATH + 'val.txt',
    'test': DATA_PATH + 'val.txt',
}
TRAIN_RGB_MEAN = [0.485, 0.456, 0.406] # Imagenet
TRAIN_RGB_SD = [0.229, 0.224, 0.225]   # Imagenet
"""

FINETUNE = True
MODEL_TYPE = 'densenet'
# alexnet / squeezenet / vgg / resnet / densenet
MODEL_DEPTH_OR_VERSION = 169

SAVE_DIR = '/workspace/wangqiushi/pytorch/projects/d8_mix/xr_normal/'
SAVED_LOGS = SAVE_DIR + 'logs/'
SAVED_MODELS = SAVE_DIR + 'saved_models/'
SAVED_PREFIX = MODEL_TYPE + str(MODEL_DEPTH_OR_VERSION) + '-full-'
TEST_MODEL_PATH = SAVE_DIR + 'densenet169-full-20181107022629.pth'
