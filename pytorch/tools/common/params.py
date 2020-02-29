MODE = 'TRAIN'
#MODE = 'TEST'

MULTI_GPU = False
GPU_DEVICE = 0 # single use
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
GPU_COUNT = len(GPU_LIST)

EPOCHS = 30

BATCHSIZE = 64
# LR = 0.0001
# LR_STEP = 10
LR = 0.01
LR_STEP = 10
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
DATA_PATH = '/data02/wangqiushi/datasets/digestive8/'
IMAGE_DIR = {
    'train': DATA_PATH + 'val/',
    'valid': DATA_PATH + 'val/',
    'test': DATA_PATH + 'test/',
}
LABEL_FILES = {
    'train': DATA_PATH + 'val.txt',
    'valid': DATA_PATH + 'val.txt',
    'test': DATA_PATH + 'test.txt',
}
TRAIN_RGB_MEAN = [0.5835075779815229, 0.34801369344856836, 0.28106787981796433] # digestive8
TRAIN_RGB_SD = [0.11965080359974416, 0.08724743240200583, 0.07926250478615704]   # digestive8
"""

# kaggle_DR
DATA_ROOT = '/raid/data/wangqiushi/kaggle/diabetic_retinopathy/'
IMAGE_ROOT = {
    'train': DATA_ROOT+'train_test/',
    'valid': DATA_ROOT+'train_test/',
    'test': DATA_ROOT+'train_test/',
}
LABEL_PATH = {
    'train': DATA_ROOT + 'labels/811_mix/train_5.csv',
    'valid': DATA_ROOT + 'labels/811_mix/valid.csv',
    'test': DATA_ROOT + 'labels/811_mix/test.csv',
}
TRAIN_RGB_MEAN = [0.485, 0.456, 0.406] # Imagenet
TRAIN_RGB_SD = [0.229, 0.224, 0.225]   # Imagenet

"""
# Replay-Attack
DATA_PATH = '/u2t/wangqiushi/datasets/RA/'
IMAGE_DIR = {
    'train': DATA_PATH+'shot_images/',
    'valid': DATA_PATH+'shot_images/',
    'test': DATA_PATH+'shot_images/',
}
LABEL_FILES = {
    'train': DATA_PATH + 'raw_train.txt',
    'valid': DATA_PATH + 'raw_valid.txt',
    'test': DATA_PATH + 'raw_test.txt',
}
TRAIN_RGB_MEAN = [0.485, 0.456, 0.406] # Imagenet
TRAIN_RGB_SD = [0.229, 0.224, 0.225]   # Imagenet
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
"""
# google_landmarks
DATA_PATH = '/data02/wangqiushi/datasets/kaggle/google_landmarks/datasets/'
IMAGE_DIR = {
    'train': DATA_PATH + 'train/',
    'valid': DATA_PATH + 'train/',
    'test': DATA_PATH + 'train/',
}
LABEL_FILES = {
    'train': DATA_PATH + 'labels/train.txt',
    'valid': DATA_PATH + 'labels/train.txt',
    'test': DATA_PATH + 'labels/train.txt',
}
TRAIN_RGB_MEAN = [0.485, 0.456, 0.406] # Imagenet
TRAIN_RGB_SD = [0.229, 0.224, 0.225]   # Imagenet
"""

"""
METRICS = {'test_acc': {},
           'confusion_matrix': {},
           'report': {},
           'cohen_kappa_score': {'weights': 'quadratic'},
}
"""

METRICS = {'test_acc': {},
           'cohen_kappa_score': {'weights': 'quadratic'},
}

LOG = {'test_acc': {},
       'cohen_kappa_score': {'weights': 'quadratic'},
}

# SAVE_BEST = ['loss', 'min']
# SAVE_BEST = ['accuracy', 'max']
SAVE_BEST = ['cohen_kappa_score', 'max']

FINETUNE = True
MODEL_TYPE = 'vgg'
# alexnet / squeezenet / vgg / resnet / densenet
MODEL_NAME = 'vgg13_bn'

SAVE_DIR = '/raid/code/wangqiushi/pycode/pytorch/projects/kaggle_mix/'
SAVED_LOGS = SAVE_DIR + 'logs/'
SAVED_MODELS = SAVE_DIR + 'saved_models/'
SAVED_PREFIX = MODEL_NAME  + '-train_1-orig-'

TEST_MODEL_PATH = SAVED_MODELS + 'resnext50_32x4d-shf-20200212070016.pth'
