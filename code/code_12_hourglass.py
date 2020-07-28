
"""
Created on Sat May 18 06:26:38 2019
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
@参考https://github.com/see--/keras-centernet
#论文https://arxiv.org/pdf/1904.07850.pdf
"""

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import numpy as np


# 定义函数，实现带卷积的BN层
def _conv_bn(_inter, filters, kernel_size, name, strides=(1, 1), padding='valid'):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                  padding=padding, use_bias=False, name=name[0])(_inter)
    bn = BatchNormalization(epsilon=1e-5, name=name[1], renorm=True)(conv)
    return bn


def residual(_x, out_dim, name, stride=1):  # 定义残差结构
    shortcut = _x
    num_channels = K.int_shape(shortcut)[-1]
    # 3*3的same卷积
    _x = ZeroPadding2D(padding=1, name=name + '.pad1')(_x)
    _x = _conv_bn(_x, out_dim, 3, strides=stride, name=[name + '.conv1', name + '.bn1'])

    _x = Activation('relu', name=name + '.relu1')(_x)
    _x = _conv_bn(_x, out_dim, 3, padding='same', name=[name + '.conv2', name + '.bn2'])

    if num_channels != out_dim or stride != 1:  # 按照步长进行下采样
        shortcut = _conv_bn(shortcut, out_dim, 1, strides=stride, name=[name + '.shortcut.0', name + '.shortcut.1'])

    _x = Add(name=name + '.add')([_x, shortcut])  # 将特征数据相加，形成残差结构
    _x = Activation('relu', name=name + '.relu')(_x)
    return _x


def convolution(_x, k, out_dim, name, stride=1):  # 定义函数，实现带有补0的卷积层（same）
    padding = (k - 1) // 2
    _x = ZeroPadding2D(padding=padding, name=name + '.pad')(_x)
    _x = _conv_bn(_x, out_dim, k, strides=stride, name=[name + '.conv', name + '.bn'])
    _x = Activation('relu', name=name + '.relu')(_x)
    return _x


def pre(_x, num_channels):  # 定义函数，对原始图片进行两次下采样
    _x = convolution(_x, 7, 128, name='pre.0', stride=2)  # 用大卷积核（7）进行下采样
    _x = residual(_x, num_channels, name='pre.1', stride=2)  # 用残差网络进行下采样
    return _x


# 定义函数，处理原数据特征，作为右侧特征
def right_features(bottom, hgid, dims):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8, f16, f32 : 1, 1/2, 1/4 1/8, 1/16, 1/32 resolution
    # 5 times reduce/increase: (256, 384, 384, 384, 512)
    features = [bottom]
    for kk, nh in enumerate(dims):  # 按照指定维度，对嵌套沙漏原数据进行特征处理
        pow_str = ''
        for _ in range(kk):
            pow_str += '.center'
        _x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, pow_str), stride=2)
        _x = residual(_x, nh, name='kps.%d%s.down.1' % (hgid, pow_str))
        features.append(_x)
    return features


# 定义函数，将左侧的原数据特征与右侧的沙漏特征相加，并返回
def connect_left_right(right, left, num_channels, num_channels_next, name):
    # left: 2 residual modules
    right = residual(right, num_channels_next, name=name + 'skip.0')
    right = residual(right, num_channels_next, name=name + 'skip.1')

    # up: 2 times residual & nearest neighbour
    out = residual(left, num_channels, name=name + 'out.0')
    out = residual(out, num_channels_next, name=name + 'out.1')
    out = UpSampling2D(name=name + 'out.upsampleNN')(out)
    out = Add(name=name + 'out.add')([right, out])
    return out


def bottleneck_layer(_x, num_channels, hgid):  # 用4个残差层进行特征处理
    # 4 residual blocks with 512 channels in the middle
    pow_str = 'center.' * 5
    _x = residual(_x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
    _x = residual(_x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
    _x = residual(_x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
    _x = residual(_x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
    return _x


def left_features(rightfeatures, hgid, dims):
    lf = bottleneck_layer(rightfeatures[-1], dims[-1], hgid)  # 对最里侧的沙漏进行特征处理
    for kk in reversed(range(len(dims))):  # 按照嵌套沙漏的顺序进行左右特征的残差叠加
        pow_str = ''
        for _ in range(kk):
            pow_str += 'center.'
        lf = connect_left_right(rightfeatures[kk], lf, dims[kk], dims[max(kk - 1, 0)],
                                name='kps.%d.%s' % (hgid, pow_str))
    return lf


def create_heads(heads, lf1, hgid):  # 定义函数，创建输出节点
    _heads = []
    for head in sorted(heads):
        num_channels = heads[head]
        _x = Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(lf1)
        _x = Activation('relu', name=head + '.%d.0.relu' % hgid)(_x)
        # 用1*1的卷积进行维度变换，生成指定通道的特征
        _x = Conv2D(num_channels, 1, use_bias=True, name=head + '.%d.1' % hgid)(_x)
        _heads.append(_x)
    return _heads


def hourglass_module(heads, bottom, cnv_dim, hgid, dims):  # 按照指定的维度进行嵌套沙漏模型的搭建
    rfs = right_features(bottom, hgid, dims)  # 按照指定维度，创建右侧特征


    lf1 = left_features(rfs, hgid, dims)  # 将右侧的特征与左侧的沙漏结构相加，得到最终特征
    lf1 = convolution(lf1, 3, cnv_dim, name='cnvs.%d' % hgid)  # 对最终特征进行卷积

    # add 1x1 conv with two heads, inter is sent to next stage
    # head_parts is used for intermediate supervision
    heads = create_heads(heads, lf1, hgid)  # 生成沙漏的输出节点
    return heads, lf1


# 实现沙漏模型结构
def HourglassNetwork(heads, num_stacks, cnv_dim=256, inputsize=(512, 512),
                     dims=[256, 384, 384, 384, 512]):
    # 定义输入节点
    input_layer = Input(shape=(inputsize[0], inputsize[1], 3), name='HGInput')
    inter = pre(input_layer, cnv_dim)
    prev_inter = None
    outputs = []
    for i in range(num_stacks):  # 按照指定的沙漏模块进行叠加
        prev_inter = inter
        _heads, inter = hourglass_module(heads, inter, cnv_dim, i, dims)
        outputs.extend(_heads)
        if i < num_stacks - 1:  # 如果后面还有沙漏，则进行残差处理后输入下个沙漏

            inter_ = _conv_bn(prev_inter, cnv_dim, 1, name=['inter_.%d.0' % i, 'inter_.%d.1' % i])

            cnv_ = _conv_bn(inter, cnv_dim, 1, name=['cnv_.%d.0' % i, 'cnv_.%d.1' % i])

            inter = Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)

    model = Model(inputs=input_layer, outputs=outputs)# 组合成模型并返回
    return model


if __name__ == '__main__':
    kwargs = {
        'num_stacks': 2,
        'cnv_dim': 256,
        'inres': (512, 512),

    }
    heads = {
        'hm': 80,
        'reg': 2,
        'wh': 2
    }
    model = HourglassNetwork(heads=heads, **kwargs)
    print(model.summary(line_length=200))
    print(model.outputs)
