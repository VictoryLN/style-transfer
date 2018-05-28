from LossCalculator import LossCalculator
import ImgProcess
import argparse
import tensorflow as tf
from PIL import Image
import settings as st
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='video input')
    parser.add_argument('-c', '--content',
                        default='Resources/ContentImages/default_content.jpg',
                        help='content image input')
    parser.add_argument('-s', '--style',
                        default='Resources/StyleImages/default_style.jpg',
                        help='style image input')
    return parser.parse_args()  # 返回一个参数字典


def train():

    # pre_process
    img_mats = ImgProcess.get_img_mat(content_path, style_path)
    content_mat = img_mats['content']
    style_mat = img_mats['style']

    # random noised img
    noise = np.random.normal(loc=0, scale=1/256, size=(1, st.HEIGHT, st.WIDTH, 3))
    rand_mat = noise * st.NOISE_RATE + (1-st.NOISE_RATE) * content_mat
    rand_mat = np.clip(rand_mat, -1, 1).astype(np.float32)
    rand_img = tf.Variable(dtype=tf.float32, initial_value=rand_mat)
    print(rand_img.shape)
    loss_calculator = LossCalculator(content_mat, style_mat)
    cost = loss_calculator.loss(rand_img)
    print(cost)
    train_steps = tf.train.AdamOptimizer(st.STUDY_RATE).minimize(cost)
    # VGG model
    with tf.Session() as sess:
        print('session begin')
        sess.run(tf.global_variables_initializer())
        print('init')
        for i in range(0, st.ITERATIONS):
            loss = sess.run(cost)
            sess.run(train_steps)
            print('train_steps')
            # if i % st.MESSAGE_TIME-1:
            print('After {} time(s) iteration, loss:{}'.format(i, loss))
            if i % st.MESSAGE_TIME == 0:
                generated_mat = sess.run(rand_img)
                ImgProcess.out_mat_img(generated_mat)
    # Loss.test(img_mats)
    # print(content_path)
    # print(img_mats['content'])
    # get loss
    # loss = Loss.getLoss(imgMat)
    # optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
    # with tf.Session() as sess:
    #     sess.run(optimizer)


if __name__ == '__main__':
    args = parse_args()
    if args.video:
        print("Start video style transfer")
    else:
        print("Start image style transfer")
        content_path = args.content
        style_path = args.style
        print(content_path)
        train()
