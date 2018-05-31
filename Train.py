from LossCalculator import LossCalculator
import ImgProcess
import tensorflow as tf
import settings as st
import numpy as np


def train(content_path, style_path, gen_size):

    # pre_process
    print('pre-process')
    img_info = ImgProcess.get_img_info(content_path, style_path, gen_size)
    content_mat = img_info['content']['mat']
    style_mat = img_info['style']['mat']
    content_name = img_info['content']['name']
    style_name = img_info['style']['name']
    content_features = img_info['content']['features']
    style_grams = img_info['style']['grams']
    generated_name = img_info['generated']['name']
    save_mat = img_info['generated']['past_save']
    # print(generated_name)
    print('pre-process done')
    # get saved or random noised img
    print('try to use checkpoint')
    if save_mat is None:
        noise = np.random.normal(loc=0, scale=1, size=(1, gen_size[1], gen_size[0], 3))
        rand_mat = noise * st.NOISE_RATE + (1-st.NOISE_RATE) * content_mat
        # rand_mat = content_mat
        # rand_mat = np.clip(rand_mat, -255, 255).astype(np.float32)
        print('no checkpoint,random noised image')
    else:
        rand_mat = save_mat
        print('load checkpoint')
    rand_img = tf.Variable(dtype=tf.float32, initial_value=rand_mat)
    # print(rand_img.shape)

    # loss calculator
    print('Initialize loss calculator')
    if content_features is None or style_grams is None:
        loss_calculator = LossCalculator(content_name, style_name,
                                         content_mat=content_mat, style_mat=style_mat)
    else:
        loss_calculator = LossCalculator(content_name, style_name,
                                         content_feature=content_features, style_grams=style_grams)
    print('loss calculator done')
    # loss
    cost = loss_calculator.loss(rand_img)
    # print(cost)

    # optimizer
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(st.STUDY_RATE, global_step, 50, 0.96, staircase=True)
    train_steps = tf.train.AdamOptimizer(st.STUDY_RATE).minimize(cost)

    # train
    with tf.Session() as sess:
        print('session begin')
        sess.run(tf.global_variables_initializer())
        print('initialize done')
        for i in range(st.ITERATIONS):
            loss = sess.run(cost)
            sess.run(train_steps)
            # print('train_steps')
            # if i % st.MESSAGE_TIME-1:
            print('After {} time(s) iteration, loss:{}'.format(i, loss))
            if i % st.MESSAGE_TIME == st.MESSAGE_TIME-1:
                generated_mat = sess.run(rand_img)
                ImgProcess.out_mat_img(generated_mat, generated_name)
