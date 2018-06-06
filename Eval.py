import Train
import argparse
import settings as st
import io_process as io_p
from NetModel import GEN
import NetModel
import tensorflow as tf
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='video input')
    parser.add_argument('-c', '--content',
                        default='Resources/ContentImages/default_content.jpg',
                        help='content image input')
    parser.add_argument('-s', '--style',
                        default='Models/default_style.ckpt',
                        help='style image checkpoint path')
    parser.add_argument('-sz', '--size',
                        default=(st.WIDTH, st.HEIGHT),
                        help='generated image size(WxH)')
    parser.add_argument('-n', '--save_name',
                        default=st.DEFAULT_SAVE_NAME,
                        help='save img name')
    return parser.parse_args()  # 返回一个参数字典


def main():
    args = parse_args()
    if args.video:
        print("Start video style transfer")
    else:
        print("Start image style transfer")
        content_path = args.content
        ckpt_path = args.style
        size = args.size
        save_name = args.save_name
        print(content_path)
        test_img = io_p.new_get_test_img(content_path)
        gen = GEN().buildNet(test_img)
        with tf.Session() as sess:
            io_p.load_ckpt(sess, ckpt_path)
            generated_img = sess.run(gen)
            print(generated_img)
            io_p.save_img(generated_img, save_name, sess)
#        Train.train(content_path, style_path, 'generated.jpg', size)


if __name__ == '__main__':
    main()
