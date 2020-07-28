
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
Created on Thu Mar  7 14:55:44 2019
"""
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


# STN转换层
class STNtransformer(tf.keras.layers.Layer):

    def __init__(self, output_size, **kwargs):  # 初始化
        self.output_size = output_size
        super(STNtransformer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):  # 输出形状
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, inputtensors, mask=None):  # 调用方法
        X, transformation = inputtensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _transform(self, X, affine_transformation, output_size):  # 转换方法
        num_channels = X.shape[-1]
        batch_size = K.shape(X)[0]
        # 将转换参数变为[2,3]矩阵
        transformations = tf.reshape(affine_transformation, shape=(batch_size, 2, 3))
        # 根据输出大小生成原始坐标(batch_size, 3, height * width)
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        print('regular_grids', K.shape(regular_grids))
        # 在原始坐标上按照转换参数进行仿射变换，生成映射坐标(batch_size, 2, height * width)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        # 根据映射坐标从原始图片上取值并填充到目标图片
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        # 设置目标图片的形状
        interpolated_image = tf.reshape(
            interpolated_image, tf.stack([batch_size, output_size[0], output_size[1], num_channels]))
        return interpolated_image

    def _make_regular_grids(self, batch_size, height, width):  # 根据输出大小生成原始坐标
        # 按照目标图片尺寸，生成坐标（所有坐标值域都在[-1,1]之间）
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)
        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = K.flatten(x_coordinates)
        y_coordinates = K.flatten(y_coordinates)
        # 组成3列矩阵，最后一列填充1
        ones = tf.ones_like(x_coordinates)
        grid = tf.concat([x_coordinates, y_coordinates, ones], 0)

        # 支持批次操作，按照批次复制原始坐标
        grid = K.flatten(grid)
        grids = K.tile(grid, K.stack([batch_size]))
        return tf.reshape(grids, (batch_size, 3, height * width))

    def _interpolate(self, image, sampled_grids, output_size):  # 根据坐标获取像素值
        batch_size = K.shape(image)[0]
        height = K.shape(image)[1]
        width = K.shape(image)[2]
        num_channels = K.shape(image)[3]
        # 取出映射坐标
        x = tf.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = tf.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')
        # 还原映射坐标对应原图的值域，由[-1,1]到[0,width]和[0,height]
        x = .5 * (x + 1.0) * tf.cast(width, dtype='float32')
        y = .5 * (y + 1.0) * tf.cast(height, dtype='float32')
        # 将转化后的坐标变为整数，同时算出相邻坐标
        x0 = K.cast(x, 'int32')
        x1 = x0 + 1
        y0 = K.cast(y, 'int32')
        y1 = y0 + 1

        # 截断出界的坐标
        max_x = int(K.int_shape(image)[2] - 1)
        max_y = int(K.int_shape(image)[1] - 1)
        x0 = K.clip(x0, 0, max_x)
        x1 = K.clip(x1, 0, max_x)
        y0 = K.clip(y0, 0, max_y)
        y1 = K.clip(y1, 0, max_y)

        # 适配批次处理
        pixels_batch = K.arange(0, batch_size) * (height * width)
        pixels_batch = K.expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = K.repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = K.flatten(base)  # 批次中每个图片的起始索引

        # 计算4个点在原图上的索引
        base_y0 = base + (y0 * width)
        base_y1 = base + (y1 * width)
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        # 将原图展开，所有批次的图片都连在一起
        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        # 按照索引取值
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')
        # 计算4个点的有效区域
        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)

        
        
        # 按照区域大小对像素加权求和
        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d


if __name__ == '__main__':  # 测试STN层

    import imageio
    import matplotlib.pyplot as plt

    im = imageio.imread(r'./girl.jpg')
    plt.figure( figsize=(12,9) )
    plt.imshow(im)
    plt.show()
    
    im = im / 255.
    im = im.reshape(1, 800, 600, 3)
    im = im.astype('float32')
    sampling_size = (400, 300)

    dense1 = tf.keras.layers.Dense(6, kernel_initializer='zeros',
                                   bias_initializer=tf.keras.initializers.constant(
                                       [[0.5, 0, 0.1], [0, 0.5, -0.5]]))  # ([[1.,0,0],[0,1.,0]]))

    locnet = tf.zeros([1, 800 * 600 * 3])
    locnet = dense1(locnet)
    print(locnet) 

    x = STNtransformer(sampling_size)([im, locnet])

    plt.imshow( (x.numpy()[0]*255).astype(np.uint8) )
    plt.show()


