import numpy as np
import tensorflow as tf
import scipy.io as sio
import settings
# vgg19.mat
# 'layers' 1 * 43 cell
# cell: for conv
# 1)weights
# 1 * 2 m_cell
# m_cell:kernels,biases
#
# 2)pad[1, 1, 1, 1]
#
# 3)type
# 'conv'
#
# 4)name
# 'conv1_1'
#
# 5)stride[1, 1]
# cell for relu
# 1)type 'relu'
# 2)name 'relu 1_1'


class VGG19(object):
    def __init__(self):
        self.vgg19 = sio.loadmat(settings.VGG19_PATH)
        self.mean = self.vgg19['normalization'][0][0][0][0][0]
    # def __vgg19_mat_test(self):
    #     vgg19 = sio.loadmat(settings.VGG19_PATH)
    #     print("vgg19['layers'].shape = ", vgg19['layers'].shape)
    #     # output: (1,43)
    #     # --wow, it a matrix
    #     # use vgg19['layer'] to get the layer matrix
    #     print("vgg19['layers'][0].shape = ", vgg19['layers'][0].shape)
    #     # output: (43,)
    #     # --means the first row of layer matrix
    #     # use vgg19['layers'][0] to get 43 cells lists(including all the parameters in 43 structure--conv,relu,pool,fc,prob)
    #     print("vgg19['layers'][0][i].shape = ", vgg19['layers'][0][0].shape)
    #     # output: (1,1)
    #     # --it means we get a matrix again. a cell is a (1,1) matrix
    #     # use vgg19['layers'][0][i] to get one cell matrix
    #     print("vgg19['layers'][0][0][0].shape = ", vgg19['layers'][0][0][0].shape)
    #     # output: (1,)
    #     # ---emm, the first row again,but it's not a new structure(I prefer to call it--core). it's a cores list
    #     # use vgg19['layers'][0][i][0] to get a core list
    #     print("vgg19['layers'][0][0][0][0].shape = ", vgg19['layers'][0][0][0][0].shape)
    #     # output: ()
    #     # --ooo! Finally, we get a core, it's not matrix. we need deeply test to know what it is.
    #     # use vgg19['layers'][0][0][0][0] to get a core
    #     print("len(vgg19['layers'][0][0][0][0]) = ", len(vgg19['layers'][0][0][0][0]))
    #     # output:
    #     # --means it is a list or a turple
    #     print("vgg19['layers'][0][1][0][0] = ", vgg19['layers'][0][1][0][0])
    #     # output:(array(['relu'], dtype='<U4'), array(['relu1_1'], dtype='<U7'))
    #     # --The core stores it parameters,like (type,name).
    #     # --As for conv layer,it also stores weights(including kernels and bias),padding and stride.
    #     # --And the weights is in the first place
    #     print("vgg19['layers'][0][1][0][0]['type'] = ", vgg19['layers'][0][41][0][0]['type'])
    #     # output:['relu']
    #     # --So the ...['type'][0] is its layer type
    #     # --...['type'][0] can be used to part conv,relu,pool,fc and prob
    #     print("vgg19['layers'][0][0][0][0][0].shape = ", vgg19['layers'][0][0][0][0][0].shape)
    #     # output:(1,2)
    #     # --the conv weights is a matrix.
    #     print("vgg19['layers'][0][0][0][0][0][0].shape = ", vgg19['layers'][0][0][0][0][0][0].shape)
    #     # output:(2,)
    #     # --so the kernels is vgg19['layers'][0][0][0][0][0][0][0].
    #     # --And bias is vgg19['layers'][0][0][0][0][0][1].WTF!!!
    #     kernels, bias = vgg19['layers'][0][0][0][0][0][0]
    #     print("kernels:\n", kernels, "\nbias:\n", bias)
    #     print("kernels.shape = ", kernels.shape)
    #     # output:(3,3,3,64)
    #     # --In VGGMatrix the convnet weights are [width,height,in_channels, out_channels]
    #     # --In tensorflow the convnet weights are [height,width,in_channels, out_channels]
    #     # --So we need to transpose VGGMatrix to (1,0,2,3)----swap the first and second dimension
    #     print("bias.shape = ", bias.shape)
    #     # output: (1,64)
    #     # --a bias matrix
    #     print(bias.reshape(-1))
    #     # output: [... ... ...]
    #     # --get the bias

    def build_net(self, input_mat):
        print('build net')
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        cells = self.vgg19['layers'][0]
        cur_lay = input_mat
        net = {}
        for i, name in enumerate(layers):
            weight = cells[i][0][0]
            # for cell in cells:
            #     weight = cell[0][0]
            # cur_type = weight['type'][0]
            # print("cur_type:", cur_type)
            # name = weight['name'][0]
            # if cur_type == 'conv': wrong, cuz fc.type == 'conv'
            # print('Building', name)
            if name[:4] == 'conv':
                kernels, bias = weight['weights'][0]
                # print('get kernels.shape:', kernels.shape, '\nbias.shape:', bias.shape)
                kernels = np.transpose(a=kernels, axes=(1, 0, 2, 3))
                # print('After transpose, kernels.shape:', kernels.shape, '\nbias.shape:', bias.shape)
                bias = bias.reshape(-1)
                # print('bias is reshaped')
                cur_lay = self.__conv_layer(cur_lay, kernels, bias, name)
            elif name[:4] == 'pool':
                cur_lay = self.__pool_layer(cur_lay, name)
            elif name[:4] == 'relu':
                cur_lay = self.__relu_layer(cur_lay, name)
            net[name] = cur_lay
            # print(name, 'is constructed')
            # print('')
        # print('done')
        return net

    def __conv_layer(self,input_x, kernels, biases, name):
        conv = tf.nn.conv2d(input=input_x, filter=tf.constant(kernels), strides=[1, 1, 1, 1], padding='SAME', name=name)
        bias = tf.nn.bias_add(conv, biases)
        return bias

    def __pool_layer(self,input_x, name):
        # Pool stride is [1, 2, 2, 1],same as ksise. BUG
        pool2 = tf.nn.max_pool(value=input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool2

    def __relu_layer(self,input_x, name):
        return tf.nn.relu(features=input_x, name=name)


# still in coding
class GEN(object):
    def __init__(self, weights=None):
        self.weights = weights
        if weights is None:
            # random generated weights
            pass
            self.weights = None
        # then use weights build net

    def buildNet(self, input_mat, training=False):
        # input_mat should be zero-mean

        pass

    def res_block(self, input_x, kernels, bias, strides, training=False):
        # kernels saved the next two conv's kernels
        k1,k2 = kernels
        b1,b2 = bias
        conv_1 = self.conv(input_x, k1, b1, strides, training)
        conv_2 = self.conv(conv_1, k2, b2, strides, training)
        return input_x+conv_2
        pass

    def pool(self):
        pass

    def conv(self, input_x, kernels, bias, stride=1, training=False):
        strides = [1, stride, stride, 1]
        if training is False:
            conv = tf.nn.conv2d(input=input_x, filter=tf.Variable(kernels),
                                strides=strides, padding='SAME')
            bias = tf.nn.bias_add(conv, tf.Variable(bias))
        else:
            conv = tf.nn.conv2d(input=input_x, filter=tf.constant(kernels),
                                strides=strides, padding='SAME')
            bias = tf.nn.bias_add(conv, tf.constant(bias))
        return bias