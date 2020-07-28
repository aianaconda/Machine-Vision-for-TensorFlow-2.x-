
"""
Created on Wed Sep 11 11:08:36 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""

import glob
import tensorflow as tf
import numpy as np
import cv2


print("TensorFlow : {}".format(tf.__version__))
classnum = {'black': 0, 'white': 1}  # 数据分类


class MakeTfdataset(object):
    # tf-dataset数据制作，初始化加载self.dataroot目录中的数据，可以使用read_data转换为tf-dataset数据

    def __init__(self):
        self.dataroot = './dataset'  # 定义数据集路径
        self.X_data = []  # 人脸图片
        self.Y_data = []  # 人脸图片对应的标签，black（0）和white（1）
        self.write_data()  # 把数据存入到X_data,Y_data

    def load_image(self, addr, shape=(32, 32)):
        #
        img = cv2.imread(addr)  # 根据路径读取图片
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        return img

    def write_data(self):
        # 遍历图片路径和图片标签并存入到X_data,Y_data
        for i in classnum.keys():
            images = glob.glob(self.dataroot + '/' + str(i) + '/*.jpg')
            labels = int(classnum[i])
            print(labels, '\t\t', i)
            for img in images:
                img = self.load_image(img)
                self.X_data.append(img)
                self.Y_data.append(labels)

    def read_data(self):  # 图片数组转换为tf-dataset数据
        self.X_data = np.array(self.X_data)
        self.Y_data = np.array(self.Y_data)
        dx_train = tf.data.Dataset.from_tensor_slices(self.X_data)
        dy_train = tf.data.Dataset.from_tensor_slices(self.Y_data).map(lambda z: tf.one_hot(z, len(classnum)))

        train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(50000).repeat().batch(256).prefetch(
            tf.data.experimental.AUTOTUNE)
        return train_dataset


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
class Net(object):  # 网络设计 -- 训练

    def __init__(self):
        # 加载tf.dataset数据
        self.M = MakeTfdataset()
        self.train_dataset = self.M.read_data()

        self.saver_root = './weights/'  # 定义路径，用于保存模型
        self.build_model()  # 建立网络
        # print(self.next_element)

    def build_model(self):
        # 建立网络-loss-优化器-准确率
        self.model = Sequential([
            Conv2D(32, (3, 3),  input_shape=(32, 32, 3)),
            LeakyReLU(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPool2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(rate=0.2),  # 将20%的节点丢弃
            Dense(2, activation='softmax')
        ])

        # 编译模型，指定模型的优化方法、损失函数、度量等
        self.model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(1e-4),
                           loss='mse', metrics=['accuracy'])

    def load_model(self):
        if tf.train.latest_checkpoint(self.saver_root) != None:
            # 加载模型的权值
            self.model.load_weights(self.saver_root + 'my_model')

    def train(self, epochs=None, saver=False, steps_per_epoch=30):
        # 训练模型，指定训练数据、训练的epochs
        self.model.fit(self.train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
        if saver:
            # 保存模型的权值到外部文件
            self.model.save_weights(self.saver_root + 'my_model')


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    N = Net()  # 加载模型
    N.train(20, True)
    # 打印模型结构
    N.model.summary()

    # 模型可视化
    import os

    os.environ["PATH"] += os.pathsep + r'D:/download/graphviz-2.38/release/bin/'
    import tensorflow as tf

    img = tf.keras.utils.plot_model(N.model, to_file="model.png", show_shapes=True)


