import tensorflow as tf
import settings as st
import os



def get_test_img(img_path):
    try:
        read_byt = tf.read_file(img_path)
        img = tf.image.decode_jpeg(read_byt,channels=3)
        img = tf.image.resize_images(img, [474,712])
        img = tf.expand_dims(img, 0)
        img = tf.cast(img, tf.float32)
        img = img/255.0
        with tf.Session() as sess:
            img = sess.run(img)
    except:
        img = None
        print('img is None')
    return img

def get_train_img(img_path): # one batch
    try:
        # Tensor: type uint8 , shape [height, width, num_channels]
        with tf.device('/cpu:0'):
            print('get raw img from', img_path)
            read_byt = tf.read_file(st.TRAIN_SET_DIR + img_path)
            img = tf.image.decode_jpeg(read_byt, channels=3) # tf.image.decode_images 不能返回tensor
            img = process(img)
            print(img)
            print('get raw img done')
    except:
        img = None
        print('img is None')
    return img

def new_get_test_img(img_path): # one batch
    try:
        read_byt = tf.read_file(img_path)
        img = tf.image.decode_jpeg(read_byt,channels=3)
        img = tf.image.resize_images(img, [474, 712])
        img = tf.expand_dims(img, 0)
        img = tf.cast(img, tf.float32)
        img = img/255.0
    except:
        img = None
        print('img is None')
    return img

def process(img):
    #  Input: 3-D tensor
    #  Output: 3-D tensor
    target_size = [st.HEIGHT,st.WIDTH]
    img = tf.image.resize_images(img, target_size)
    img = tf.cast(img, tf.float32)
    return img


def get_style_img(img_path):
    try:
        # Tensor: type uint8 , shape [height, width, num_channels]
        with tf.device('/cpu:0'):
            print('get img from', img_path)
            read_byt = tf.read_file(img_path)
            img = tf.image.decode_jpeg(read_byt, channels=3)
            target_height = st.HEIGHT
            target_width = st.WIDTH
            # img = tf.image.resize_image_with_crop_or_pad(img, target_height, target_width)
            # 这两者不能换位置
            if img.get_shape().ndims == 3:
                img = tf.expand_dims(img, 0)
                # print(img)
            img = tf.cast(img, tf.float32)
            img = img - st.MEAN_PIXEL
            print('get processed style img')
    except:
        img = None
        print('img is None')
    print(img.get_shape())

    return img


def get_img_batch_iterator(img_dir, batch_size, epoch=2, shuffle=True):
    filelist = os.listdir(img_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filelist)
    dataset = dataset.map(get_train_img)
    # map 在batch之前，不然map是以一整个batch作为运算
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(shuffle)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    return next_batch


def unprocess(img):
    return tf.clip_by_value(tf.cast(img + st.MEAN_PIXEL, tf.uint8), 0, 255)


def save_img(img, name,sess):
    with tf.device('/cpu:0'):
        print('save generated image')
        img = tf.clip_by_value(tf.cast(img,tf.uint8),0,255)
        img = tf.squeeze(img)
        out_byt = tf.image.encode_jpeg(img)
        save_op = tf.write_file(st.GENERATED_PATH + name, out_byt)
        sess.run(save_op)
        print('save success')


def save_raw_img(img,name,sess):
    with tf.device('/cpu:0'):
        print('save raw image')


def save_ckpt(sess, save_ckpt_path):
    saver = tf.train.Saver()
    saver.save(sess, save_ckpt_path)
    print('Model saved in path %s' % save_ckpt_path)


def load_ckpt(sess, ckpt_path):
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    print('model restored.')

