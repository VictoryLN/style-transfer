from PIL import Image
import settings as st
import numpy as np
import scipy.misc as sc


def get_img_info(content_path, style_path):
    print("Processing.py receives content path: ", content_path)
    # check picture exist
    if not (content_path.endswith('.jpg') and style_path.endswith('.jpg')):
        print("only support picture type:jpg")
    # get name
    content_name = content_path[content_path.rindex('/')+1:content_path.rindex('.')]
    style_name = style_path[style_path.rindex('/')+1:style_path.rindex('.')]
    generated_name = content_name + '_' + style_name
    # import img
    print('importing images')
    content_img = Image.open(content_path)  # R G B
    if content_img is None:
        print("Not found such content image")
    style_img = Image.open(style_path)
    if style_img is None:
        print("Not found such style image")
    print('importing images done')
    # resize img
    print('resize images')
    size = (st.WIDTH, st.HEIGHT)

    content_img = content_img.resize(size, Image.BILINEAR)
    # box = (200,200,424,424)
    # style_img = style_img.crop(box)
    # style_img = style_img.resize(size, Image.BILINEAR)
    # sc.imsave('{}{}.jpg'.format(st.GENERATED_PATH, 'normal'), style_img)
    print('resize images done')
    # content_img.show()
    # style_img.show()

    # transfer to matrix
    content_mat = np.asarray(a=content_img, dtype=np.float32)
    style_mat = np.asarray(a=style_img, dtype=np.float32)

    # [0,255] zero-mean and normalized
    content_mat = content_mat - st.MEAN_PIXEL
    style_mat = style_mat - st.MEAN_PIXEL
    # content_mat = content_mat / 256
    # style_mat = style_mat / 256
    # transfer to 1*N*N*3
    # dim = (1, st.HEIGHT, st.WIDTH, 3)
    #
    # content_mat = np.reshape(a=content_mat, newshape=dim)
    # style_mat = np.reshape(a=style_mat, newshape=dim)
    # print(content_mat.shape)
    # print(content_mat)
    content_mat = np.expand_dims(a=content_mat, axis=0)
    style_mat = np.expand_dims(a=style_mat, axis=0)
    print(style_mat.shape)
    # get features, grams and past result
    print('try to get saved')
    content_features = {}
    style_grams = {}
    try:
        for layer in st.CONTENT_LAYER:
            content_features[layer] = np.load(st.SAVE_CONTENT_PATH + content_name + '_' + layer + '.npy')
        for layer in st.STYLE_LAYERS:
            style_grams[layer] = np.load(st.SAVE_STYLE_PATH + style_name + '_' + layer+'.npy')
        past_save = np.load(st.SAVE_GENERATED_PATH + generated_name + '.npy')
        print('success')
    except:
        content_features = None
        style_grams = None
        past_save = None
        print('failed')
    img_info = {'content': {'mat': content_mat, 'name': content_name, 'features': content_features},
                'style': {'mat': style_mat, 'name': style_name, 'grams': style_grams},
                'generated': {'name': generated_name, 'past_save': past_save}}
    return img_info


def out_mat_img(img_mat, img_name):
    print('output and save generated image')
    np.save(st.SAVE_GENERATED_PATH + img_name + '.npy', img_mat)
    img_mat = img_mat[0]
    img_mat += st.MEAN_PIXEL
    img_mat = np.clip(img_mat, 0, 255).astype(np.uint8)
    sc.imsave('{}{}.jpg'.format(st.GENERATED_PATH, img_name), img_mat)
    print('done')
