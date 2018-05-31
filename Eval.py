import Train
import argparse
import settings as st

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='video input')
    parser.add_argument('-c', '--content',
                        default='Resources/ContentImages/default_content.jpg',
                        help='content image input')
    parser.add_argument('-s', '--style',
                        default='Resources/StyleImages/default_style.jpg',
                        help='style image input')
    parser.add_argument('-sz', '--size',
                        default=(st.WIDTH, st.HEIGHT),
                        help='generated image size(WxH)')
    return parser.parse_args()  # 返回一个参数字典


def main():
    args = parse_args()
    if args.video:
        print("Start video style transfer")
    else:
        print("Start image style transfer")
        content_path = args.content
        style_path = args.style
        size = args.size
        print(content_path)
        Train.train(content_path, style_path, size)


if __name__ == '__main__':
    main()
