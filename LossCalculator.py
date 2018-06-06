from NetModel import VGG19
from NetModel import GEN
import NetModel
import tensorflow as tf
import settings as st
import numpy as np
import io_process as io_p

class LossCalculator(object):
    def __init__(self, style_img, style_weight, content_weight,tv_weight):
        # Input: preprocessed style_img
        self.vgg = VGG19()
        self.gen = GEN()
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.style_grams = self.get_style_grams(style_img)
        self.tv_weight = tv_weight

    def get_style_grams(self, style_img):
        # Input : preprocessed style_img in shape [1,h,w,3]
        # Brief:  compute style grams
        # Attention: all grams are divided by size at last
        # Output: style_grams dict{layer:gram}

        # g = tf.Graph()
        # with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        style_grams = {}
        with tf.Session() as sess:
            net = self.vgg.build_net(style_img)
            for layer in st.STYLE_LAYERS:
                features = sess.run(net[layer])
                # features shape [1,h,w,c] to [x,c]
                features = tf.reshape(features, [-1, features.shape[3]])
                size = features.shape[0].value * features.shape[1].value
                # size = tf.shape(features).value()[0] * tf.shape(features).value()[1]
                print('style gram size:',size)
                gram = sess.run(tf.matmul(features, features, transpose_a=True) / size)  # gram求出来要除size
                print('gram=', gram)
                style_grams[layer] = gram
        return style_grams

    def __del__(self):
        print('LossCalculator Destroyed')

    def loss(self, input_batch):
        # Input : a batch of preprocessed content_img in shape [batch,h,w,3] ndarray
        # 1. build gen input_batch to get generate_output
        # 2. process the output
        # 2. build vgg of input_batch and generate_output to get content_features and generate_featuers
        # 3. style_loss(generater_features)
        # 4. content_loss(content_features,generater_features)
        # 5. total variation
        # 6. add all
        # Output : total_loss tensor
        # build gen
        print('input_batch', input_batch)
        pre_gen_output = input_batch - st.MEAN_PIXEL
        gen_output = self.gen.buildNet(pre_gen_output)
        print('gen_output', gen_output)
        # process the output
        processed_gen_output = gen_output - st.MEAN_PIXEL
        print('gen_output - mean', gen_output)
        # build vgg net
        gen_net = self.vgg.build_net(processed_gen_output)
        content_net = self.vgg.build_net(input_batch)
        print('gen_net', gen_net)
        print('content_net',content_net)
        # content_loss
        content_loss = 0.0
        for layer in st.CONTENT_LAYER:
            _, height, width, channels = map(lambda i: i.value, content_net[layer].get_shape())
            size = height * width * channels
            content_loss += tf.nn.l2_loss(gen_net[layer] - content_net[layer]) * 2 / tf.to_float(size)
        content_loss = content_loss * self.content_weight
        # style_loss
        style_loss = 0.0
        for layer in st.STYLE_LAYERS:
            gram, size = self.__gram(gen_net[layer])
            style_loss += tf.nn.l2_loss(gram - self.style_grams[layer]) * 2 / tf.to_float(size)  # l2_loss([batch,g] - [g])
        style_loss = style_loss * self.style_weight
        # total_variation_loss
        # tv_loss = self.total_variation_loss(gen_output)

        # total_loss = content_loss * self.content_weight + style_loss * self.style_weight + tv_loss * self.tv_weight
        total_loss = content_loss + style_loss
        return total_loss, content_loss, style_loss, gen_output

    def total_variation_loss(self, img):
        shape = img.get_shape()
        height = shape[1].value
        width = shape[2].value
        y = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(img, [0, 1, 0, 0],
                                                                                            [-1, -1, -1, -1])
        x = tf.slice(img, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(img, [0, 0, 1, 0],
                                                                                           [-1, -1, -1, -1])
        loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
        return loss


    def __gram(self, features):
        # Input : 4-D tensor shape [batch,h,w,c]
        # Output: 2-D tensor shape [batch,g]
        batch = features.shape[0].value
        # print('batch',batch)
        height = features.shape[1].value
        weight = features.shape[2].value
        # print('weight',weight)
        # print('height',height)
        channel = features.shape[-1].value
        features = tf.reshape(features, [-1,weight*height, channel])
        # print(features.get_shape())
        size = features.shape[1].value * features.shape[2].value  # feature.shape[1]是Dimension类型，不是int类型
        print('gen gram size:',size)
        gram = tf.matmul(features, features, transpose_a=True) / size
        # print(gram.get_shape())
        return gram, size

