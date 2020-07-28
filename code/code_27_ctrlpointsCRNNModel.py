"""
Created on Wed Sep 11 11:08:36 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
# 导入基础模块
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
# 导入本工程的其它模块
from code_20_CRNNModel import FeatureExtractor, CRNN
from code_26_ctrlpointsLayers import SpatialTransformer


# 定义函数，实现抽取控制点
def ctrlpointsFeatureExtractor(x, keypoint, init_bias):
    x = FeatureExtractor(x)  # 提取图片特征
    conv_output = GlobalMaxPooling2D()(x)  # 全局最大池化，输出的形状为[batch,256]
    # 两层全连接网络，实现控制点的回归
    fc1 = Dense(128, activation='relu')(conv_output)
    fc2 = Dense(2 * keypoint, kernel_initializer='zeros', activation='relu',
                bias_initializer=tf.keras.initializers.constant(init_bias)
                )(0.1 * fc1)
    # 改变控制点形状并返回
    return Reshape((keypoint, 2))(ctrl_pts)


def CPCRNNctc(model_config):  # 定义函数，实现使用控制点的矫正模型
    # 定义输入节点
    input_tensor = Input((model_config['tagsize'][0], model_config['tagsize'][1], model_config['ch']))
    # 初始化控制点
    init_bias = build_init_bias(keypoint=20, activation=None, pattern='identity',
                                margins=(0.01, 0.01))
    # 提取控制点
    input_control_points = ctrlpointsFeatureExtractor(input_tensor, keypoint=20, init_bias=init_bias)
    # 实例化控制点对象
    cp_transformer = SpatialTransformer(
        output_image_size=model_config['tagsize'],
        num_control_points=20,
        margins=(0.0, 0.0)
    )
    # 按照控制点对图片空间变换
    x = cp_transformer([input_tensor, input_control_points])
    x = Reshape((model_config['tagsize'][0], model_config['tagsize'][1], model_config['ch']))(x)

    # 对变换后的图片用CRNN模型处理
    CRNN_model = CRNN(model_config)
    y_pred = CRNN_model(x)
    # 将各个网络层连起来，组合层模型
    CPCRNN_model = Model(inputs=input_tensor, outputs=y_pred, name='CPCRNN_model')
    return CPCRNN_model
