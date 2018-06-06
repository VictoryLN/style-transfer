WIDTH = 224
HEIGHT = 224
VGG19_PATH = 'Resources/VGG-19/imagenet-vgg-verydeep-19.mat'
DEFAULT_STUDY_RATE = 0.001
DEFAULE_BATCH_SIZE = 4
NOISE_RATE = 0.00
DEFAULT_ITERATIONS = 40000
MESSAGE_COUNTER = 500
DEFAULT_SAVE_NAME = 'default_save.jpg'
DEFAULT_STYLE_WEIGHT = 250
DEFAULT_TV_WEIGHT = 0.0
DEFAULT_CONTENT_WEIGHT = 1
DEFAULT_STYLE_JPEG = 'Resources/StyleImages/default_style.jpg'
DEFAULT_SAVE_PATH = './Models/'
TRAIN_SET_DIR = '../../train2014/'
CONTENT_LAYER = ('relu4_2',)
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
GENERATED_PATH = './Output/40000-instancenormal-250D1-zeromean-scream/'
SAVE_GENERATED_PATH = './Save/generated/'
SAVE_STYLE_PATH = './Save/style/'
SAVE_CONTENT_PATH = './Save/content/'
MEAN_PIXEL = [123.68, 116.779, 103.939]
# CONTENT_WEIGHT = 5e0
# STYLE_WEIGHT = 1e2
# TV_WEIGHT = 1e2
# LEARNING_RATE = 1e-2
# STYLE_SCALE = 1.0
# ITERATIONS = 1000



# net model gen output* 255 - Mean