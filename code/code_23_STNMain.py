"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
Created on Thu Mar  7 14:55:44 2019
"""

from code_22_STNLayers import STNtransformer
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

dataset_path = "./datasets/mnist_cluttered_60x60_6distortions.npz"

data = np.load(dataset_path)  # 加载数据集
for key in data.keys():  # 显示数据集中的内容
    print(key, np.shape(data[key]))

# 制作训练数据集
traindataset = tf.data.Dataset.from_tensor_slices((data['x_train'], data['y_train']))


def _mapfun(x, y):  # 定义函数，对每个样本进行变形操作
    x = tf.reshape(x, [60, 60, 1])
    return x, y


# 制作测试数据集
testdataset = tf.data.Dataset.from_tensor_slices((data['x_test'], data['y_test'])).map(_mapfun)
# 制作验证数据集
vailiddataset = tf.data.Dataset.from_tensor_slices((data['x_valid'], data['y_valid'])).map(_mapfun)

# 显示数据集中的内容
for one in traindataset.take(1):
    img = np.reshape(one[0], (60, 60))
    print(one[1], img.shape)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray', interpolation='none')
    strtitle = ' MNIST sample:%d' % tf.argmax(one[1]).numpy()
    plt.title(strtitle, fontsize=40)
    plt.axis('off')
    plt.show()


############################STNmodel

def IC(inputs, p):  # 定义独立组件层
    # 带有renorm的BN层
    x = BatchNormalization(renorm=True)(inputs)
    return Dropout(p)(x)  # 按照p的百分比进行丢弃节点


def CNN(x):  # 带有IC层的卷积神经网络
    x = Conv2D(512, 5, strides=3, padding='same', activation='relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = IC(x, 0.2)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = IC(x, 0.2)
    return x


def STNmodel(input_shape=(60, 60, 1), sampling_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape)
    x = CNN(image)
    x = Conv2D(20, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    locnet = GlobalAveragePooling2D()(x)  # 【batch，20】
    # 生成仿射变换参数
    locnet = Dense(6, kernel_initializer='zeros',  # activation='tanh',
                   bias_initializer=tf.keras.initializers.constant([[1., 0, 0], [0, 1., 0]]))(locnet)
    # 进行仿射变换
    x = STNtransformer(sampling_size, name='STNtransformer')([image, locnet])
    x = CNN(x)  # 对仿射变换后的图片进行分类处理
    x = Conv2D(10, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)  # 注意点 10
    x = GlobalAveragePooling2D()(x)  # 【batch，10】
    x = Activation('softmax')(x)  # 注意点配合categorical_crossentropy
    return Model(inputs=image, outputs=x)  # 生成模型并返回


model = STNmodel()  # 实例化模型对象
model.compile(loss='categorical_crossentropy', optimizer='adam')  # keras.losses.categorical_crossentropy

#  定义函数用于显示训练结果
def print_evaluation(epoch_arg, val_score, test_score):
    message = 'Epoch: {0} | ValLoss: {1} | TestLoss: {2}'
    print(message.format(epoch_arg, val_score, test_score))


num_epochs = 11  # 定义数据集迭代训练的次数
batch = 64  # 定义批次
# 制作训练数据集
traindataset = traindataset.shuffle(buffer_size=len(data['y_train'])).batch(
    batch, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# 制作验证和测试数据集
vailiddataset = vailiddataset.batch(64, drop_remainder=False)
testdataset = testdataset.batch(64, drop_remainder=False)
for epoch in range(num_epochs):  # 按照指定迭代次数进行训练
    for dataone in traindataset:  # 遍历数据集
        img = np.reshape(dataone[0], (batch, 60, 60, 1))
        loss = model.train_on_batch(img, dataone[1])
    print(loss)
    if epoch % 10 == 0:  # 每迭代2次数据集显示一次训练结果
        val_score = model.evaluate(vailiddataset.take(20), verbose=0)  # (*val_data, verbose=1)
        test_score = model.evaluate(testdataset.take(20), verbose=0)  # (*test_data, verbose=1)
        print_evaluation(epoch, val_score, test_score)  # 输出训练结果
        print('-' * 40)  # 输出分割线


# 定义函数，以9宫格形式，可视化结果
def plot_mnist_grid(image_batch, function=None):
    fig = plt.figure(figsize=(6, 6))
    # 取出9个数据
    if function is not None:
        image_result = function([image_batch[:9]])
    else:
        image_result = np.expand_dims(image_batch[:9], 0)
    plt.clf()  # 清空缓存
    # 设置子图间距离
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for image_arg in range(9):  # 依次将图片显示到9宫格中
        plt.subplot(3, 3, image_arg + 1)
        image = np.squeeze(image_result[0][image_arg])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    fig.canvas.draw()
    plt.show()


# 将STN层取出
input_image = model.input
output_STN = model.get_layer('STNtransformer').output  # 通过model.summary()找到该层名字
STN_function = K.function([input_image], [output_STN])
# 显示原始数据
plot_mnist_grid(img)
# 显示STN变换后的数据
plot_mnist_grid(img, STN_function)
# 输出预测结果
out = model.predict([img[:9]])
print('预测结果：', tf.argmax(out, axis=1).numpy())
