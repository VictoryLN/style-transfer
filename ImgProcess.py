from PIL import Image
import settings
import numpy as np
import scipy.misc as sc


def get_img_mat(content_path, style_path):
    print("Processing.py receives content path: ", content_path)
    # import img
    if not content_path.endswith('.jpg'):
        print("only support picture type:jpg")
    content_img = Image.open(content_path)
    if content_img is None:
        print("Not found such content image")
    style_img = Image.open(style_path)
    if style_img is None:
        print("Not found such style image")

    # resize img
    size = (settings.WIDTH, settings.HEIGHT)
    content_img = content_img.resize(size)
    style_img = style_img.resize(size)
    content_img.show()
    style_img.show()

    # transfer to matrix
    content_mat = np.asarray(a=content_img, dtype=np.float32)
    style_mat = np.asarray(a=style_img, dtype=np.float32)

    # [0,255] map to [-1,1]
    content_mat = content_mat - 127.5
    style_mat = style_mat - 127.5
    content_mat = content_mat / 128
    style_mat = style_mat / 128

    # transfer to 1*N*N*3
    dim = (1, settings.HEIGHT, settings.WIDTH, 3)
    content_mat = np.reshape(a=content_mat, newshape=dim)
    style_mat = np.reshape(a=style_mat, newshape=dim)
    # print(content_mat.shape)
    # print(content_mat)
    img_mats = {'content': content_mat, 'style': style_mat}
    # img_mats = (content_mat, style_mat) 为了清楚起见，还是用字典
    return img_mats


def out_mat_img(img_mat):
    img_mat = img_mat * 128
    img_mat += [128.0, 128.0, 128.0]
    img_mat = img_mat[0]
    img_mat = np.clip(img_mat, 0, 255).astype(np.uint8)
    sc.imsave('{}.jpg'.format(settings.GENERATED_NAME), img_mat)
    print('Output success!')
