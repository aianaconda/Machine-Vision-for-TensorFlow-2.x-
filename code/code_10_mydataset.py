# -*- coding: utf-8 -*-
"""
Created on Sat May 18 06:26:38 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""



# 显示dataset中的样本标注
# 引入基础模块
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import array_ops
from distutils.version import LooseVersion
from code_09_mydataset import isNewAPI,get_img_lab_files



def get_dataset(config, shuffle=True):
    img_lab_files = get_img_lab_files(config['gt_path'], config['image_path'])

    def _parse_function(filename, config):  # 定义图像解码函数
        image_string = tf.io.read_file(filename[0])
        if isNewAPI == True:
            img = tf.io.decode_jpeg(image_string, channels=3)  # 1.13, 2.0
        else:
            img = tf.image.decode_jpeg(image_string, channels=3)  # 1.12
        # 获取图片形状
        h, w, c = array_ops.unstack(array_ops.shape(img), 3)
        img = tf.reshape(img, [h, w, c])
        #        print("sss", h, w, img.shape)  # 直接通过shape拿不到形状-

        if isNewAPI == True:
            img = tf.image.resize(img, config['tagsize'], tf.image.ResizeMethod.BILINEAR)
        else:
            img = tf.image.resize_images(img, config['tagsize'], tf.image.ResizeMethod.BILINEAR)

        x_sl = config['tagsize'][1] / tf.cast(w, dtype=tf.float32)  # 宽-----------------
        y_sl = config['tagsize'][0] / tf.cast(h, dtype=tf.float32)  # 高--------------

        def my_py_func(x_sl, y_sl, filename):  # -----------

            if isNewAPI == True:  # 新的API里传入的是张量
                x_sl, y_sl, filename = x_sl.numpy(), y_sl.numpy(), filename.numpy()  # ---------

            # 打开文件，读取标注
            filestr = open(filename, "r", encoding='utf-8')
            filelines = filestr.readlines()
            filestr.close()

            Y = []
            if len(filelines) != 0:
                for i, line in enumerate(filelines):
                    if 'img' in filename.decode():  # 文件有两种格式，一种是空格分割，一种是逗号分割
                        file_data = line.split(', ')
                    else:
                        file_data = line.split(' ')
                    if len(file_data) <= 4:
                        continue

                    # 计算变形后，区域框的坐标
                    xmin = int(file_data[0]) * x_sl  # ------------
                    xmax = int(file_data[2]) * x_sl
                    ymin = int(file_data[1]) * y_sl
                    ymax = int(file_data[3]) * y_sl

                    Y.append([xmin, ymin, xmax, ymax, 1])
            return np.asarray(Y, dtype=np.float32)

            # 构建自动图进行处理

        if isNewAPI == True:
            threefun = tf.py_function
        else:
            threefun = tf.py_func
        ground_truth = threefun(my_py_func,
                                [x_sl, y_sl, filename[1]], tf.float32)  # ------------

        return img, ground_truth

    dataset = tf.data.Dataset.from_tensor_slices(img_lab_files)

    if shuffle == True:
        dataset = dataset.shuffle(buffer_size=len(img_lab_files))
    dataset = dataset.map(lambda x: _parse_function(x, config))

    # 根据一批次进行填充
    dataset = dataset.padded_batch(config['batchsize'], padded_shapes=([None, None, None], [None, None]),
                                   drop_remainder=True)
    if isNewAPI == True:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(1)

    return dataset


# 从标签中提取出区域框坐标，和分类概率
def getboxesbythreshold(gt, threshold):
    gtmask = gt[gt[..., -1] > threshold]  # 过滤掉无效的gt
    scores = gtmask[..., -1]
    boxes = np.int32(gtmask[..., 0:4])
    return boxes, scores


# 根据图片、区域坐标、分类概率合成到一个图片中，进行显示
def showimgwithbox(img, boxes, scores, y_first=False):  # gt=[x,y,x2,y2,score]-------------
    from PIL import ImageDraw
    color = tuple(np.random.randint(0, 256, 3))

    draw = ImageDraw.Draw(img)  # 定义Draw对象，用于合成图片
    for i in range(len(boxes)):
        if y_first == True:  # 支持两种坐标格式
            box = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])  # [y,x,y2,x2]
        else:
            box = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])  # [x,y,x2,y2]
        draw.rectangle(xy=box, outline=color, width=4)  # 在Draw对象上画矩形框
        # 在指定区域显示文本
        plt.text(box[0], box[1], str(scores[i]), color='red', fontsize=12)
    plt.imshow(img)  # 显示合成好的图片
    plt.show()


if __name__ == '__main__':
    from PIL import Image

    assert LooseVersion(tf.__version__) >= LooseVersion("2.0")  # 2.0以下需要手动打开动态图

    dataset_config = {
        'batchsize': 4,
        'image_path': r'data/test/images/',
        'gt_path': r'data/test/ground_truth/',
        'tagsize': [384, 384]
    }

    # 制作图片数据集
    image_dataset = get_dataset(dataset_config, shuffle=False)

    for i, gt in image_dataset.take(1):  # 取出1批次数据
        i = i[0]  # 取出批次中第1个图片
        # 取出批次中第1个图片所对应的标注，形状为[x,y,x2,y2,score]
        gt = gt[0].numpy()
        img = Image.fromarray(np.uint8(i.numpy()))

        # 将图片与标注合成到一起显示出来
        boxes, scores = getboxesbythreshold(gt, 0)
        showimgwithbox(img, boxes, scores)
