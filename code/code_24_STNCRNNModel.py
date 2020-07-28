
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
# 导入其它模块
from code_20_CRNNModel import FeatureExtractor, CRNN
from code_22_STNLayers import STNtransformer


# 定义函数，封装STN层
def STNChange(x, sampling_size, inputimg):
    locnet = GlobalAveragePooling2D()(x)  # 全局平均池化后的形状为[batch，20]
    print('locnet', locnet.get_shape())

    locnet = Dense(6, kernel_initializer='zeros',
                   bias_initializer=tf.keras.initializers.constant([[1., 0, 0], [0, 1., 0]]))(locnet)

    # 仿射变化
    x = STNtransformer(sampling_size)([inputimg, locnet])
    return x


def STNCRNN(model_config):
    input_tensor = Input((model_config['tagsize'][0], model_config['tagsize'][1], model_config['ch']))
    x = FeatureExtractor(input_tensor)  # 提取图片特征

    sampling_size = (model_config['tagsize'][0], model_config['tagsize'][1])

    x = STNChange(x, sampling_size, input_tensor)  # 仿射变换

    # 将各个网络层连起来，组合层模型
    CRNN_model = CRNN(model_config)
    y_pred = CRNN_model(x)
    STNCRNN_model = Model(inputs=input_tensor, outputs=y_pred, name="STNCRNN_model")
    return STNCRNN_model
