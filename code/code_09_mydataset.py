# -*- coding: utf-8 -*-
"""
Created on Sat May 18 06:26:38 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
"""

# 用dataset读取数据

# 引入基础
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import array_ops
from distutils.version import LooseVersion

isNewAPI = True
if LooseVersion(tf.__version__) >= LooseVersion("1.13"):
    print("new version API")
else:
    print('old version API')
    isNewAPI = False


def get_img_lab_files(gt_path, image_path):  # 定义函数，获取文件名
    img_lab_files = []
    # 在不同的平台，顺序不一样。sorted很必要。方便调试时，统一数据
    dirlist = sorted(os.listdir(gt_path))
    print(dirlist[:6])
    for new_file in dirlist:
        name_split = new_file.split('.')
        image_name = name_split[0][3:]
        image_name = image_name + '.jpg'
        if 'gt' in new_file:
            image_name = name_split[0][3:]
            image_name = image_name + '.jpg'
        img_file = os.path.join(image_path, image_name)
        lab_file = os.path.join(gt_path, new_file)
        img_lab_files.append((img_file, lab_file))
    return img_lab_files


def get_dataset(config, shuffle=True):  # 制作数据集
    # 获取样本文件和标签的列表
    img_lab_files = get_img_lab_files(config['gt_path'], config['image_path'])

    def _parse_function(filename, config):  # 定义图像解码函数

        # 读取图片
        image_string = tf.io.read_file(filename[0])
        if isNewAPI == True:
            img = tf.io.decode_jpeg(image_string, channels=3)  # 适用于 1.13, 2.0及之后的版本
        else:
            img = tf.image.decode_jpeg(image_string, channels=3)  # 适用于 1.12及之前的版本

        # 获取图片形状
        h, w, c = array_ops.unstack(array_ops.shape(img), 3)
        img = tf.reshape(img, [h, w, c])

        # 对图片尺寸进行调整
        if isNewAPI == True:
            img = tf.image.resize(img, config['tagsize'], tf.image.ResizeMethod.BILINEAR)
        else:
            img = tf.image.resize_images(img, config['tagsize'], tf.image.ResizeMethod.BILINEAR)

        def my_py_func(filename):  # 定义函数，用于制作标签

            if isNewAPI == True:  # 新的API里传入的是张量，需要进行转换
                filename = filename.numpy()

            # 打开文件，读取样本的标注文件
            filestr = open(filename, "r", encoding='utf-8')
            filelines = filestr.readlines()
            filestr.close()

            Y = []
            if len(filelines) != 0:
                for i, line in enumerate(filelines):
                    # 标注文件中有两种格式，一种是空格分割，一种是逗号分割
                    if 'img' in filename.decode():
                        file_data = line.split(', ')
                    else:
                        file_data = line.split(' ')
                    if len(file_data) <= 4:  # 排除无效的行
                        continue

                    # 计算变形后，区域框的坐标
                    xmin = int(file_data[0])
                    xmax = int(file_data[2])
                    ymin = int(file_data[1])
                    ymax = int(file_data[3])
                    Y.append([xmin, ymin, xmax, ymax, 1])
            return np.asarray(Y, dtype=np.float32)

        # 构建自动图，处理样本标签
        if isNewAPI == True:
            threefun = tf.py_function
        else:
            threefun = tf.py_func
        ground_truth = threefun(my_py_func,
                                [filename[1]], tf.float32)

        return img, ground_truth  # 将处理过的样本图片和标签返回

    # 将列表数据转成数据集形式
    dataset = tf.data.Dataset.from_tensor_slices(img_lab_files)

    if shuffle == True:  # 将样本的顺序打乱
        dataset = dataset.shuffle(buffer_size=len(img_lab_files))

    # 对每个样本进行再加工
    dataset = dataset.map(lambda x: _parse_function(x, config))


    # 根据设定的批次大小，对数据集进行填充
    dataset = dataset.padded_batch(config['batchsize'], padded_shapes=([None, None, None], [None, None]),
                                   drop_remainder=True)

    # 设置数据集的缓存大小
    if isNewAPI == True:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(1)
    return dataset


if __name__ == '__main__':
    from PIL import Image

    # 2.0以下需要手动打开动态图
    assert LooseVersion(tf.__version__) >= LooseVersion("2.0")

    dataset_config = {
        'batchsize': 4,
        'image_path': r'data/test/images/',
        'gt_path': r'data/test/ground_truth/',
        'tagsize': [384, 384]
    }

    # 图片数据集
    image_dataset = get_dataset(dataset_config, shuffle=False)

    for i, j in image_dataset.take(1):
        i = i[0]
        j = j[0]
        print(i)
        img = Image.fromarray(np.uint8(i.numpy()))
        plt.imshow(img)  # 用于显示
        plt.show()
        print(j)
