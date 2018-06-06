from LossCalculator import LossCalculator
import tensorflow as tf
import settings as st
import io_process as io_p
import argparse
from NetModel import GEN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sw', '--style_weight',
                        default=st.DEFAULT_STYLE_WEIGHT,
                        help='style weight')
    parser.add_argument('-cw', '--content_weight',
                        default=st.DEFAULT_CONTENT_WEIGHT,
                        help='content weight')
    parser.add_argument('-tw', '--tv_weight',
                        default=st.DEFAULT_TV_WEIGHT,
                        help='total variation weight')
    parser.add_argument('-s', '--style',
                        default=st.DEFAULT_STYLE_JPEG,
                        help='style image input')
    parser.add_argument('-tsdir', '--train_set_dir',
                        default=st.TRAIN_SET_DIR,
                        help='train images directory')
    parser.add_argument('-p', '--save_path',
                        default=st.DEFAULT_SAVE_PATH,
                        help='save checkpoint path')
    parser.add_argument('-i', '--iteration',
                        default=st.DEFAULT_ITERATIONS,
                        help='iteration')
    parser.add_argument('-r', '--study_rate',
                        default=st.DEFAULT_STUDY_RATE,
                        help='study rate')
    parser.add_argument('-b', '--batch_size',
                        default=st.DEFAULE_BATCH_SIZE,
                        help='batch size')
    parser.add_argument('-e', '--checkpoint_exist',
                        help='checkpoint exist,continue training')
    return parser.parse_args()  # 返回一个参数字典


def train(style_path, style_weight, content_weight, tv_weight, iteration, study_rate, batch_size, save_ckpt_path, exist):
    style_name = style_path[style_path.rindex('/')+1:style_path.rindex('.')]
    if not save_ckpt_path.endswith('/'):
        save_ckpt_path = save_ckpt_path + '/'
    save_ckpt_path = save_ckpt_path + style_name + '.ckpt'
    # get preprocessed img
    content_placeholder = tf.placeholder(dtype=tf.float32,shape=[batch_size, st.HEIGHT, st.WIDTH, 3])
    style_img = io_p.get_style_img(style_path)
    # the calculator initialization will compute the style grams
    loss_cal = LossCalculator(style_img, style_weight, content_weight, tv_weight)
    next_batch = io_p.get_img_batch_iterator(st.TRAIN_SET_DIR, batch_size)
    loss, closs, sloss, gen = loss_cal.loss(content_placeholder)
    train_step = tf.train.AdamOptimizer(study_rate).minimize(loss)
    with tf.Session() as sess:
        if exist:
            print('try to load checkpoint')
            io_p.load_ckpt(sess, save_ckpt_path)  # try block is useless, if error the progress would be shutdown
        else:
            print('no checkpoint, build random variable')
            sess.run(tf.global_variables_initializer())
        try:
            for i in range(iteration):
                res=sess.run(next_batch)
                # print(res)
                fd={content_placeholder: res}
                sess.run(train_step, fd)
                print(i)
                # print('content loss = ', closs.eval())
                # print('style loss = ', sloss.eval())
                if i % st.MESSAGE_COUNTER == st.MESSAGE_COUNTER-1:
                    # test_img = io_p.new_get_test_img('Resources/ContentImages/default_content.jpg')
                    generated_img = sess.run(gen,fd)
                    io_p.save_img(res[0],'r{}.jpg'.format(i),sess)
                    io_p.save_img(generated_img[0], 'g{}.jpg'.format(i), sess)
                    print('alert')
                    # cur_loss = sess.run(loss,fd)
                    # print('{} times:loss=:{}'.format(i, cur_loss))
                    # print('content loss = ', sess.run(closs,fd))
                    # print('style loss = ', sess.run(sloss,fd))
                    io_p.save_ckpt(sess, save_ckpt_path)
        except tf.errors.OutOfRangeError:
            print('end')
            exit(0)


if __name__ == '__main__':
    args = parse_args()
    print("Start image style transfer")
    sw = args.style_weight
    cw = args.content_weight
    tw = args.tv_weight
    tsdir = args.train_set_dir
    sp = args.save_path
    s = args.style
    i = args.iteration
    r = args.study_rate
    b = args.batch_size
    e = args.checkpoint_exist
    e = False
    train(s, sw, cw, tw,
          i, r, b, sp, e)


