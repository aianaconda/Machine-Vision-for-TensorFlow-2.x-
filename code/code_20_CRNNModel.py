"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from code_19_mydataset import isNewAPI

# 根据TensorFlow不同的版本，引入不同的GRU库
if isNewAPI == True:
    from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU
else:
    from tensorflow.keras.layers import CuDNNGRU


def FeatureExtractor(x):  # 定义卷积网络，提取图片特征
    for i in range(5):
        for j in range(2):
            x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
        x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, strides=2 if i < 2 else (2, 1), padding='same',
                   activation='relu')(x)
        x = BatchNormalization()(x)
    return x


def RNNFeature(x):
    x = Permute((2, 1, 3))(x)  # 转换维度
    # 转化为适合于RNN网络的输入格式
    x = TimeDistributed(Flatten())(x)  # 32个序列

    # 定义基于GRU Cell的双向RNN网络
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    return x



def CRNN(model_config):# 定义函数，搭建CRNN模型
    #    定义模型的输入节点
    input_tensor = Input((model_config['tagsize'][0], model_config['tagsize'][1], model_config['ch']))
    x = FeatureExtractor(input_tensor)  # 提取图片特征

    x = RNNFeature(x)  # 转化成RNN特征

    # 用全连接网络实现输出层
    y_pred = Dense(model_config['outputdim'], activation='softmax')(x)

    # 在计算CTC Loss时，模型输出的序列个数32必须要大于样本标签序列个数label_len
    print('y_pred:', y_pred.get_shape())  # （batch,32,66）

    # 将各个网络层连起来，组合层模型
    CRNN_model = Model(inputs=input_tensor, outputs=y_pred, name="CRNN_model")

    return CRNN_model  # 返回CRNN模型


#定义CTC损失函数
def ctc_lambda_func(y_true, y_pred, model_config, **kwargs):  # 在2。0下没有**kwargs会编译不过

    outputstep = y_pred.get_shape()[1]  # 获得输入数据的序列长度


    # 为批次中的每个数据，单独指定序列长度
    input_length = np.asarray([[outputstep]] * model_config['batchsize'], dtype=np.int)
    label_length = np.asarray([[model_config['label_len']]] * model_config['batchsize'])
    # input_length必须大于label_length，否则会提示无效的ctc

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)