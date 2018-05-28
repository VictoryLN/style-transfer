WIDTH = 224
HEIGHT = 224
VGG19_PATH = 'Resources/VGG-19/imagenet-vgg-verydeep-19.mat'
STUDY_RATE = 0.01
NOISE_RATE = 0.05
ITERATIONS = 1000
MESSAGE_TIME = 50
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1
CONTENT_LAYER = ('relu4_2',)
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
GENERATED_NAME = 'outputImg'

