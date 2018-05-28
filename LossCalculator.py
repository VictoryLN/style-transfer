from NetModel import VGG19
import tensorflow as tf
import settings as st
import numpy as np


class LossCalculator(object):
    def __init__(self, content_mat, style_mat, content_feature=None, style_feature=None):
        self.content_tensor = tf.constant(content_mat)
        self.style_tensor = tf.constant(style_mat)
        self.vgg = VGG19()
        self.content_feature = {}
        self.style_feature = {}
        self.generated_feature = None
        self.style_gram = {}
        self.content_layer_num = len(st.CONTENT_LAYER)
        self.style_layer_num = len(st.STYLE_LAYERS)
        if content_feature is None:
            with tf.Session() as sess:
                print('\nget content features')
                # input_mat = tf.placeholder(dtype=tf.float32, shape=(1, st.HEIGHT, st.WIDTH, 3), name='input_mat')
                net = self.vgg.build_net(self.content_tensor)
                for layer in st.CONTENT_LAYER:
                    features_mat = sess.run(net[layer])
                    self.content_feature[layer] = tf.constant(features_mat)
                print('\ndone')
        if style_feature is None:
            with tf.Session() as sess:
                print('\nget style features')
                # input_mat = tf.placeholder(dtype=tf.float32, shape=(1, st.HEIGHT, st.WIDTH, 3), name='input_mat')
                net = self.vgg.build_net(self.style_tensor)
                for layer in st.STYLE_LAYERS:
                    features = sess.run(net[layer])
                    self.style_gram[layer], _, __ = self.__gram(features)
                # print(self.style_feature)
                print('\ndone')

    def __del__(self):
        print('LossCalculator Destroyed')

    def loss(self, generated_mat):
        print('computing total loss.')
        net = self.vgg.build_net(generated_mat)
        # content_loss
        content_loss = 0.0
        for layer in st.CONTENT_LAYER:
            layer_loss = tf.nn.l2_loss(net[layer] - self.content_feature[layer])
            print(layer_loss)
            content_loss = content_loss + layer_loss
        content_loss = content_loss / self.content_layer_num
        print(content_loss)
        # style_loss
        style_loss = 0.0
        for layer in st.STYLE_LAYERS:
            gram, n_l, m_l = self.__gram(net[layer])
            print(gram.shape)
            layer_loss = tf.nn.l2_loss(gram - self.style_gram[layer]) / (2 * m_l * m_l * n_l * n_l)
            print(layer_loss.shape)
            style_loss = style_loss + layer_loss
        style_loss = style_loss / self.style_layer_num

        total_loss = content_loss * st.CONTENT_WEIGHT + style_loss * st.STYLE_WEIGHT
        return total_loss

    def __gram(self, features):
        print('computing gram matrix.')
        if isinstance(features, np.ndarray):
            print('it\'s ndarray:', features.shape)
            # features 为1xHxWxN的三维数组
            features = np.reshape(features, (-1, features.shape[-1]))  # 化为(H*W)xN的矩阵
            n_l = features.shape[1]
            m_l = features.shape[0]
            # print('features.reshape', features.shape)
            gram = np.matmul(features.T, features)
            gram = tf.constant(gram)
        elif isinstance(features, tf.Tensor):
            print('it\'s tensor', features.shape)
            features = tf.reshape(features, [-1, features.shape[-1]])
            n_l = features.shape[1].value  # feature.shape[0]是Dimension类型，不是int类型
            m_l = features.shape[0].value
            features_t = tf.transpose(features, [1, 0])
            gram = tf.matmul(features_t, features)
        else:
            print('error type')
            gram = None
            n_l = None
            m_l = None
        print('gram.shape=', gram.shape)
        return gram, n_l, m_l

