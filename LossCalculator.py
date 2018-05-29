from NetModel import VGG19
import tensorflow as tf
import settings as st
import numpy as np


class LossCalculator(object):
    def __init__(self, content_name, style_name, content_mat=None, style_mat=None, content_feature=None, style_grams=None):
        # if content_feature is None and style_mat is None and content_feature is None and style_feature is None:
        #     exit(1)
        self.content_feature = content_feature
        self.style_grams = style_grams
        self.vgg = VGG19()
        self.generated_feature = None
        self.content_layer_num = len(st.CONTENT_LAYER)
        self.style_layer_num = len(st.STYLE_LAYERS)
        if content_mat is not None and style_mat is not None:
            self.content_tensor = tf.constant(content_mat)
            self.style_tensor = tf.constant(style_mat)
        if content_feature is None:
            self.content_feature = {}
            with tf.Session() as sess:
                print('get content features')
                print('build content net')
                net = self.vgg.build_net(self.content_tensor)
                print('build content net done')
                for layer in st.CONTENT_LAYER:
                    features_mat = sess.run(net[layer])
                    self.content_feature[layer] = features_mat
                    np.save(st.SAVE_CONTENT_PATH + content_name + '_' + layer + '.npy', self.content_feature[layer])
                print('get content features done')
        if style_grams is None:
            self.style_grams = {}
            with tf.Session() as sess:
                print('get style features')
                print('build style net')
                net = self.vgg.build_net(self.style_tensor)
                print('build style net done')
                for layer in st.STYLE_LAYERS:
                    features = sess.run(net[layer])
                    self.style_grams[layer], _, __ = self.__gram(features)
                    np.save(st.SAVE_STYLE_PATH + style_name + '_' + layer + '.npy', self.style_grams[layer])
                print('get style features done')

    def __del__(self):
        print('LossCalculator Destroyed')

    def loss(self, generated_mat):
        # print('computing total loss.')
        net = self.vgg.build_net(generated_mat)
        # content_loss
        content_loss = 0.0
        for layer in st.CONTENT_LAYER:
            layer_loss = tf.nn.l2_loss(net[layer] - self.content_feature[layer])
            # print(layer_loss)
            content_loss = content_loss + layer_loss
        content_loss = content_loss / self.content_layer_num
        # print(content_loss)
        # style_loss
        style_loss = 0.0
        for layer in st.STYLE_LAYERS:
            gram, n_l, m_l = self.__gram(net[layer])
            # print(gram.shape)
            layer_loss = tf.nn.l2_loss(gram - self.style_grams[layer]) / (2 * m_l * m_l * n_l * n_l)
            # print(layer_loss.shape)
            style_loss = style_loss + layer_loss
        style_loss = style_loss / self.style_layer_num

        total_loss = content_loss * st.CONTENT_WEIGHT + style_loss * st.STYLE_WEIGHT
        # print('done')
        return total_loss

    def __gram(self, features):
        # print('computing gram matrix.')
        if isinstance(features, np.ndarray):
            # print('it\'s ndarray:', features.shape)
            # features 为1xHxWxN的三维数组
            features = np.reshape(features, (-1, features.shape[-1]))  # 化为(H*W)xN的矩阵
            n_l = features.shape[1]
            m_l = features.shape[0]
            # print('features.reshape', features.shape)
            gram = np.matmul(features.T, features)
        elif isinstance(features, tf.Tensor):
            # print('it\'s tensor', features.shape)
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
        # print('gram.shape=', gram.shape)
        # print('done')
        return gram, n_l, m_l

