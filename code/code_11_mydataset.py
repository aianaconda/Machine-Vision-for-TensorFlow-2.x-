# -*- coding: utf-8 -*-
"""
Created on Sat May 18 06:26:38 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com     
"""

# 修改dataset，用于训练，gt改为中心点和高，宽
# 引入基础
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
        # print("sss", h, w, img.shape)  # 直接通过shape拿不到形状--展开
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

                    w = (xmax - xmin)  ##-----
                    h = (ymax - ymin)
                    # 计算变形后，区域框的中心点，并对其进行归一化
                    x = ((xmax + xmin) / 2)
                    y = ((ymax + ymin) / 2)

                    Y.append([y, x, h, w, 1])  ##----------
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
    dataset = dataset.map(lambda x: _parse_function(x, config))  #

    # 根据一批次进行填充
    padded_shapes = ([None, None, None], [60, None])  # 必须指定填充，不然最长的gt，再计算loss时会出错---展开 batch=1，none会训练不出来

    dataset = dataset.padded_batch(config['batchsize'], padded_shapes=padded_shapes, padding_values=(
    tf.constant(-1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)), drop_remainder=True)
    # 也可以使用默认的0来填充
    #    dataset = dataset.padded_batch(config['batchsize'],padded_shapes =padded_shapes,
    #                                   drop_remainder=True)

    if isNewAPI == True:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.prefetch(1)

    return dataset


def getboxesbythreshold(gt, threshold):
    gtmask = gt[gt[..., -1] > threshold]  # 过滤掉无效的gt
    scores = gtmask[..., -1]
    boxes = np.int32(gtmask[..., 0:4])
    return boxes, scores


def showimgwithbox(img, boxes, scores, y_first=False):  # gt=[x,y,x2,y2,score]-------------
    from PIL import ImageDraw
    color = tuple(np.random.randint(0, 256, 3))

    draw = ImageDraw.Draw(img)  # 用于显示
    for i in range(len(boxes)):
        if y_first == True:
            box = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])  # [x,y,x2,y2]
        else:
            box = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])  # [x,y,x2,y2]
        draw.rectangle(xy=box, outline=color, width=4)  # 用于显示
        plt.text(box[0], box[1], str(scores[i]), color='red', fontsize=12)
    plt.imshow(img)  # 用于显示
    plt.show()


def gen_dataset(dataset_config, shuffle=True):  # 定义函数，用于在静态图中初始化数据集
    dataset = get_dataset(dataset_config, shuffle)

    if isNewAPI == True:  # 对于TensorFlow1.13及以后版本，需要使用tf.compat.v1接口
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        init_op = iterator.initializer
    else:  # 对于TensorFlow 1.12及以前的版本，可以直接生成迭代器
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer

    return init_op, iterator

#将（中心点、高、宽）格式的标签转化为区域坐标
def centerbox2xybox(centbox):  #参数centbox的形状为[centy,centx,h,w]
    yx1 = centbox[..., 0:2] - centbox[..., 2:4] / 2.
    yx2 = centbox[..., 0:2] + centbox[..., 2:4] / 2.
    # 依次提取
    x1 = np.expand_dims(yx1[..., 1], -1)
    y1 = np.expand_dims(yx1[..., 0], -1)
    x2 = np.expand_dims(yx2[..., 1], -1)
    y2 = np.expand_dims(yx2[..., 0], -1)

    score = np.expand_dims(centbox[..., 4], -1)
    # 组合成坐标标签并返回
    return np.concatenate([x1, y1, x2, y2, score], axis=-1)

if __name__ == '__main__':
    from PIL import Image

    # 2.0以下需要手动打开动态图
    assert LooseVersion(tf.__version__) >= LooseVersion("2.0")

    dataset_config = {
        'batchsize': 2,
        'image_path': r'data/test/images/',
        'gt_path': r'data/test/ground_truth/',
        'tagsize': [384, 384]
    }

    # 制作图片数据集
    image_dataset = get_dataset(dataset_config, shuffle=False)

    for i, gt in image_dataset.take(1):
        print(i[0])
        i = i[0]
        print(gt[0].numpy(), gt[1].numpy())
        gt = gt[0].numpy()
        print(gt)

        img = Image.fromarray(np.uint8(i.numpy()))
        gt = centerbox2xybox(gt) #将标签转换成区域坐标的标签格式
        #从标签中提取坐标框和分类概率
        boxes, scores = getboxesbythreshold(gt, 0)
        showimgwithbox(img, boxes, scores)#显示带有标注的样本图片
