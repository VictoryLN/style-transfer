import tensorflow as tf
import settings as st


def get_img(img_path):
    # request name
    # file_names = [img_path]
    # filename_queue = tf.train.string_input_producer(file_names)
    # image_reader = tf.WholeFileReader()
    # _, image_file = image_reader.read(filename_queue)
    # image = tf.image.decode_jpeg(image_file)
    try:
        # Tensor: type uint8 , shape [height, width, num_channels]
        read_byt = tf.read_file(img_path)
        img = tf.image.decode_image(read_byt, channels=3)
        img = preprocess(img)
        print(1)
    except:
        img = None
        print(2)
    return img


def preprocess(img):
    target_height = st.HEIGHT
    target_width = st.WIDTH
    # print(img.shape)
    img = tf.image.resize_image_with_crop_or_pad(img, target_height, target_width)  # 这两者不能换位置
    print(img.get_shape().ndims)
    a = tf.rank(img)
    if img.get_shape().ndims == 3:
        img = tf.expand_dims(img, 0)
        print(img)
    img = tf.cast(img, tf.float32)
    print(img)
    return img - st.MEAN_PIXEL


def unprocess(img):
    return tf.clip_by_value(tf.cast(img + st.MEAN_PIXEL, tf.uint8), 0, 255)


def save_img(img, name):
    print('save generated image')
    img = unprocess(img)
    print(img.get_shape())
    out_byt = tf.image.encode_jpeg(img)
    save_op = tf.write_file(st.GENERATED_PATH + name, out_byt)
    return save_op

#
# image = get_img('D:/Study/Curricula/创新实践/大三下/main/style-transfer/Resources/ContentImages/content.jpg')
# save = save_img(image[0], '233.jpg')
# with tf.Session() as sess:
#     sess.run(save)
