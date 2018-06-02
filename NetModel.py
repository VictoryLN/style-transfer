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
        bias = tf.nn.bias_add(conv, tf.constant(biases))
        return bias

    def __pool_layer(self,input_x, name):
        # Pool stride is [1, 2, 2, 1],same as ksise. BUG
        pool2 = tf.nn.max_pool(value=input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
        return pool2

    def __relu_layer(self, input_x, name):
        return tf.nn.relu(features=input_x, name=name)


# still in coding
class GEN(object):
    def __init__(self):
        pass

    def buildNet(self, input_mat, training=False):
        # input_mat should be zero-mean
        input_mat = input_mat / 255.0 - 0.5
        net_info = {
            'conv_1': {'type': 'conv', 'kernel': 9, 'stride': 1, 'in-channel': 3, 'out-channel': 32,
                       'padding': 'SAME', 'Nonlinearity': 'ReLU'},
            'conv_2': {'type': 'conv', 'kernel': 3, 'stride': 2, 'in-channel': 32, 'out-channel': 64,
                       'padding': 'SAME', 'Nonlinearity': 'ReLU'},
            'conv_3': {'type': 'conv', 'kernel': 3, 'stride': 2, 'in-channel': 64, 'out-channel': 128,
                       'padding': 'SAME', 'Nonlinearity': 'ReLU'},
            'resBlk_4': {'type': 'residual_block', 'kernel': 3, 'stride': 1, 'in-channel': 128, 'out-channel': 128},
            'resBlk_5': {'type': 'residual_block', 'kernel': 3, 'stride': 1, 'in-channel': 128, 'out-channel': 128},
            'resBlk_6': {'type': 'residual_block', 'kernel': 3, 'stride': 1, 'in-channel': 128, 'out-channel': 128},
            'resBlk_7': {'type': 'residual_block', 'kernel': 3, 'stride': 1, 'in-channel': 128, 'out-channel': 128},
            'resBlk_8': {'type': 'residual_block', 'kernel': 3, 'stride': 1, 'in-channel': 128, 'out-channel': 128},
            'upsampling_9': {'type': 'upsampling', 'kernel': 3, 'stride': 1, 'in-channel': 128, 'out-channel': 64},
            'upsampling_10': {'type': 'upsampling', 'in-channel': 64, 'out-channel': 32},
            'conv_11': {'type': 'conv', 'kernel': 9, 'stride': 1, 'in-channel': 32, 'out-channel': 3,
                        'padding': 'SAME', 'Nonlinearity': 'tanh'}
        }

        x = tf.pad(input_mat, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFELCT')
        for layer in net_info.values():
            layer_type = layer['type']
            if layer_type == 'conv':
                # conv-batchNormal-relu
                x = self.conv(x, layer['kernel'], layer['in-channel'], layer['out-channel'],
                              layer['stride'])
                x = self.batch_norm(x, layer['out-channel'], training)
                if layer['Nonlinearity'] == 'relu':
                    x = self.relu(x)
                else:
                    x = self.tanh(x)
            elif layer_type == 'residual_block':
                x = self.res_block(x, layer['kernel'], layer['in-channel'], layer['out-channel'],
                                   layer['stride'])
            elif layer_type == 'upsampling':
                x = self.upsampling(x, layer['kernel'], layer['in-channel'], layer['out-channel'],
                                    layer['stride'], training)
        x = (x + 1) * 127.5
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        x = tf.image.crop_to_bounding_box(x, 10, 10, height-20, width-20)
        return x

    def res_block(self, input_x, kernels, in_channels, out_channels, stride=1):
        # kernels saved the next two conv's kernels
        k1, k2 = kernels
        conv_1 = self.conv(input_x, k1, in_channels, out_channels, stride)
        relu_1 = self.relu(conv_1)
        conv_2 = self.conv(relu_1, k2, in_channels, out_channels, stride)
        relu_2 = self.relu(conv_2)
        # add the input and the output
        return input_x+relu_2
    # Batch-Normalization---对一批数据集的某一层的输出进行归一化（减均值，除标准差），处理后均值为0，方差为1。
    # BN一般跟在CNN后面，在激活函数之前
    #
    # 偏置项没有用了（因为加入偏置项相当于偏动均值，在恢复原始特征的时候均值偏动的部分会被减去，不起任何作用）
    #

    def batch_norm(self, x, size, training, decay=0.999):
        beta = tf.Variable(tf.zeros([size]), name='beta')
        scale = tf.Variable(tf.ones([size]), name='scale')
        pop_mean = tf.Variable(tf.zeros([size]))
        pop_var = tf.Variable(tf.ones([size]))
        epsilon = 1e-3

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch_norm')

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon, name='batch_norm')

        return tf.cond(training, batch_statistics, population_statistics)

    def tanh(self, input_x):
        return tf.nn.tanh(input_x)

    def relu(self, input_x):
        pass
        # 有代码指出relu之前需要先把nan转为0，但为什么训练过程中很可能出现nan呢？
        return tf.nn.relu(input_x)

    def conv(self, input_x, kernel, in_channels, out_channels, stride=1):
        strides = [1, stride, stride, 1]
        shape = [kernel, kernel, in_channels, out_channels]
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(input=input_x, filter=weights, strides=strides, padding='SAME')
        return conv

    def img_scale(self, x, scale):
        weight = x.get_shape()[1].value
        height = x.get_shape()[2].value
        try:
            out = tf.image.resize_nearest_neighbor(x, size=(weight*scale, height*scale))
        except:
            out = tf.image.resize_images(x, size=(weight*scale, height*scale))
        return out

    def upsampling(self, input_x, kernel, in_channels, out_channels, stride=1, training=False):
        out = self.img_scale(input_x, 2)
        out = self.conv(out, kernel, in_channels, out_channels, stride)
        out = self.batch_norm(out, out_channels, training=training)
        out = self.relu(out)
        return out

