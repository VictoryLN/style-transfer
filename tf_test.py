# import tensorflow as tf
# import numpy as np
# import os
# import io_process
# import settings as st
# # file_list = os.listdir('.')
# # flist_tensor = tf.constant(file_list)
# # dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5,6]))
# # dataset = dataset.map(lambda x:x*6)
# # dataset = dataset.batch(2)
# # dataset = dataset.shuffle(1000)
# # dataset = dataset.repeat(2)
# # iterator = dataset.make_one_shot_iterator()
# # one_element = iterator.get_next() # yige batch de tu pian
# a = tf.Variable(tf.random_normal([2, 1]), name='a')
# b = tf.Variable(tf.random_normal([1]), name='b')
# # 大概一个variable[2 * 224 * 224 *3]需要1.2M
# c = a - b
#
# d = tf.nn.l2_loss(a - b)
# # def get_img_batch_iterator(img_dir, batch_size, epoch=2, shuffle=True):
# #     filelist = os.listdir(img_dir)
# #     flist_tensor = tf.constant(filelist)
# #     dataset = tf.data.Dataset.from_tensor_slices(flist_tensor)
# #     dataset = dataset.batch(batch_size)
# #     dataset = dataset.shuffle(shuffle)
# #     dataset = dataset.map(get_img)
# #     dataset = dataset.repeat(epoch)
#
# #     return next_batch
# #
# #
# # # 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
# # def _parse_function(filename):
# #     image_string = tf.read_file(filename)
# #     image_decoded = tf.image.decode_image(image_string)
# #     # image_resized = tf.image.resize_images(image_decoded, [28, 28])
# #     print(image_decoded.get_shape())
# #     return image_decoded
# #
# #
# # # 图片文件的列表
# # filenames = tf.constant(["0.1-250-15000.jpg", "outputImg.jpg"])
# # # label[i]就是图片filenames[i]的label
# #
# # # 此时dataset中的一个元素是(filename, label)
# # dataset = tf.data.Dataset.from_tensor_slices(filenames)
# #
# # # 此时dataset中的一个元素是(image_resized, label)
# # dataset = dataset.map(_parse_function)
# #
# # iterator = dataset.make_one_shot_iterator()
# #
# # next_batch = iterator.get_next()
# #
# #
#
#
# def get_img(img_path): # one batch
#     try:
#         # Tensor: type uint8 , shape [height, width, num_channels]
#         with tf.device('/cpu:0'):
#             print('get raw img from', img_path)
#             read_byt = tf.read_file(st.TRAIN_SET_DIR + img_path)
#             img = tf.image.decode_jpeg(read_byt, channels=3)  # tf.image.decode_images 不能返回tensor
#             img = process(img)
#             print('get raw img done')
#     except:
#         img = None
#         print('img is None')
#     return img
#
#
# def process(img):
#     target_size = [st.HEIGHT,st.WIDTH]
#     img = tf.image.resize_images(img, target_size)
#     img = tf.cast(img, tf.float32)
#     return img - st.MEAN_PIXEL
#
#
# def get_img_batch_iterator(img_dir, batch_size, epoch=2, shuffle=True):
#     filelist = os.listdir(img_dir)
#     dataset = tf.data.Dataset.from_tensor_slices(filelist)
#     dataset = dataset.map(get_img)
#     # map 在batch之前，不然map是以一整个batch作为运算
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.shuffle(shuffle)
#     dataset = dataset.repeat(epoch)
#     iterator = dataset.make_one_shot_iterator()
#     next_batch = iterator.get_next()
#     return next_batch
#
#
# next_batch = get_img_batch_iterator(st.TRAIN_SET_DIR, 2)
# # one_element = get_img_batch_iterator('E:/style-transfer/Resources/ContentImages/', 1)
#
# with tf.Session() as sess:
#     try:
#         print(sess.run(next_batch))
#     except tf.errors.OutOfRangeError:
#         print("end!")
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     sess.run(next_batch)
#     # try:
#     #     io_process.save_ckpt(sess,'./ckpt/111.ckpt')
#     # except:
#     #     sess.run(tf.global_variables_initializer())
#     # print('a=', a.eval())
#     # print('b=', b.eval())
#     # try:
#     #     while True:
#     #         print(sess.run(one_element))
#     # except tf.errors.OutOfRangeError:
#     #     print('end')
import tensorflow as tf
a=tf.Variable([[[[2,3,4],[4,3,2],[1,7,8],[2,2,2]],[[1,1,1],[2,3,4],[4,3,2],[9,5,1]]],[[[1,1,1],[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1],[1,1,1]]]],dtype=tf.float32)
b=tf.Variable(3)
x=tf.constant([[[2,3,4],[4,3,2],[1,7,8],[2,2,2]],[[1,1,1],[2,3,4],[4,3,2],[9,5,1]]],dtype=tf.float32)
y=tf.constant(5)
train_mean = tf.assign(a, a * x)
train_var = tf.assign(b, b * y)
batch_mean, batch_var = tf.nn.moments(a, [0, 1, 2])

#
# def f(sess):
#     with tf.control_dependencies([train_mean, train_var]):
#         return 11
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(a-x))
    # print(f(session))
    # print(train_mean.eval())
    # print(train_var.eval())
    # print(batch_mean.eval())
    # print(batch_var.eval())