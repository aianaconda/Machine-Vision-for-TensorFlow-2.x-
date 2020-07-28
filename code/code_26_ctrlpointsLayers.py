# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:08:36 2019
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
@参考：https://github.com/bgshih/aster
"""
from tensorflow.keras import backend as kb
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
import numpy as np

class SpatialTransformer(tf.keras.layers.Layer):

    def __init__(self, output_image_size=None,
                 num_control_points=None,
                 margins=None, **kwargs):

        super(SpatialTransformer, self).__init__(**kwargs)
        self._output_image_size = output_image_size
        self._num_control_points = num_control_points
        self._margins = margins
        self._output_grid = self._build_output_grid()
        self._output_ctrl_pts = self._build_output_control_points(self._margins)
        self._inv_delta_c = self._build_helper_constants()

        self._eps = 1e-6
    def compute_output_shape(self, input_shapes):
        height, width = self._output_image_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, inputtensors, mask=None):
        X, input_control_points = inputtensors
        sampling_grid = self._batch_generate_grid(input_control_points)
        rectified_images = self._batch_sample(X, sampling_grid)
        print( 'rectified_images',rectified_images.get_shape())
        return rectified_images
    def _batch_generate_grid(self, input_ctrl_pts):
        """
        Args
          input_ctrl_pts: float32 tensor of shape [batch_size, num_ctrl_pts, 2]
        Returns
          sampling_grid: float32 tensor of shape [num_sampling_pts, 2]
        """
        C = tf.constant(self._output_ctrl_pts, tf.float32)  # => [k, 2]
        batch_Cp = input_ctrl_pts  # => [B, k, 2]
        batch_size = kb.shape(input_ctrl_pts)[0]

        inv_delta_c = tf.constant(self._inv_delta_c, dtype=tf.float32)
        batch_inv_delta_c = tf.tile(tf.expand_dims(inv_delta_c, 0), [batch_size, 1, 1])  # => [B, k+3, k+3]
 
        batch_Cp_zero = tf.concat([batch_Cp, tf.zeros([batch_size, 3, 2])], axis=1)  # => [B, k+3, 2]
        batch_T = tf.matmul(batch_inv_delta_c, batch_Cp_zero)  # => [B, k+3, 2]

        k = self._num_control_points
        G = tf.constant(self._output_grid.reshape([-1, 2]), tf.float32)  # => [n, 2]

        n = kb.shape(G)[0]

        G_tile = tf.tile(tf.expand_dims(G, axis=1), [1, k, 1])  # => [n,k,2]


        C_tile = tf.expand_dims(C, axis=0)  # => [1, k, 2]
        G_diff = G_tile - C_tile  # => [n, k, 2]
        rbf_norm = tf.norm(G_diff, axis=2, ord=2, keepdims=False)  # => [n, k]
        rbf = tf.multiply(tf.square(rbf_norm), tf.math.log(rbf_norm + self._eps))  # => [n, k]
        G_lifted = tf.concat([tf.ones([n, 1]), G, rbf], axis=1)  # => [n, k+3]
        batch_G_lifted = tf.tile(tf.expand_dims(G_lifted, 0), tf.stack([batch_size, 1, 1],name='stack1'))  # => [B, n, k+3]

        batch_Gp = tf.matmul(batch_G_lifted, batch_T)

        return batch_Gp

    def _batch_sample(self, images, batch_sampling_grid):
        """
        Args:
          images: tensor of any time with shape [batch_size, image_h, image_w, depth]
          batch_sampling_grid; float32 tensor with shape [batch_size, num_sampling_pts, 2]
        """
        if images.dtype != tf.float32:
            raise ValueError('image must be of type tf.float32')
        batch_G = batch_sampling_grid

        batch_size = tf.shape(images)[0]
        image_h = tf.shape(images)[1]
        image_w = tf.shape(images)[2]
        ch = tf.shape(images)[3]

        n = tf.shape(batch_sampling_grid)[1]


        batch_Gx = tf.cast(image_w, dtype='float32') * batch_G[:, :, 0]
        batch_Gy = tf.cast(image_h, dtype='float32') * batch_G[:, :, 1]
        batch_Gx = tf.clip_by_value(batch_Gx, 0., tf.cast(image_w, dtype='float32') - 2)
        batch_Gy = tf.clip_by_value(batch_Gy, 0., tf.cast(image_h, dtype='float32') - 2)

        batch_Gx0 = tf.cast(tf.floor(batch_Gx), tf.int32)  # G* => [batch_size, n, 2]
        batch_Gx1 = batch_Gx0 + 1  # G*x, G*y => [batch_size, n]
        batch_Gy0 = tf.cast(tf.floor(batch_Gy), tf.int32)
        batch_Gy1 = batch_Gy0 + 1

        def _get_pixels(images, batch_x, batch_y, batch_indices):
            indices = kb.stack([batch_indices, batch_y, batch_x], axis=2)  # => [B, n, 3]
            pixels = tf.gather_nd(images, indices)
            return pixels

        batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1),[1, n])  # => [B, n]

        batch_I00 = _get_pixels(images, batch_Gx0, batch_Gy0, batch_indices)
        batch_I01 = _get_pixels(images, batch_Gx0, batch_Gy1, batch_indices)
        batch_I10 = _get_pixels(images, batch_Gx1, batch_Gy0, batch_indices)
        batch_I11 = _get_pixels(images, batch_Gx1, batch_Gy1, batch_indices)  # => [B, n, d]
        
        batch_Gx0 = kb.cast(batch_Gx0, tf.float32)
        batch_Gx1 = kb.cast(batch_Gx1, tf.float32)
        batch_Gy0 = kb.cast(batch_Gy0, tf.float32)
        batch_Gy1 = kb.cast(batch_Gy1, tf.float32)

        batch_w00 = (batch_Gx1 - batch_Gx) * (batch_Gy1 - batch_Gy)
        batch_w01 = (batch_Gx1 - batch_Gx) * (batch_Gy - batch_Gy0)
        batch_w10 = (batch_Gx - batch_Gx0) * (batch_Gy1 - batch_Gy)
        batch_w11 = (batch_Gx - batch_Gx0) * (batch_Gy - batch_Gy0)  # => [B, n]

        batch_pixels = tf.add_n([
            tf.expand_dims(batch_w00, axis=2) * batch_I00,
            tf.expand_dims(batch_w01, axis=2) * batch_I01,
            tf.expand_dims(batch_w10, axis=2) * batch_I10,
            tf.expand_dims(batch_w11, axis=2) * batch_I11,
        ])

        output_h, output_w = self._output_image_size
        output_maps = tf.reshape(batch_pixels, [batch_size, output_h, output_w, ch])
        output_maps = tf.cast(output_maps, dtype=images.dtype)

        return output_maps

    def _build_output_grid(self):
        output_h, output_w = self._output_image_size
        output_grid_x = (np.arange(output_w) + 0.5) / output_w
        output_grid_y = (np.arange(output_h) + 0.5) / output_h
        output_grid = np.stack(
            np.meshgrid(output_grid_x, output_grid_y),
            axis=2)
        return output_grid

    def _build_output_control_points(self, margins):
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = self._num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        output_ctrl_pts = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return output_ctrl_pts

    def _build_helper_constants(self):
        C = self._output_ctrl_pts
        k = self._num_control_points
        hat_C = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                hat_C[i, j] = np.linalg.norm(C[i] - C[j])
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((k, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, k))], axis=1)
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C




def build_init_bias( keypoint,pattern, margins, activation):
    margin_x, margin_y = margins
    num_ctrl_pts_per_side = keypoint // 2
    upper_x = np.linspace(margin_x, 1.0 - margin_x, num=num_ctrl_pts_per_side)
    lower_x = np.linspace(margin_x, 1.0 - margin_x, num=num_ctrl_pts_per_side)

    if pattern == 'slope': #斜线
        upper_y = np.linspace(margin_y, 0.3, num=num_ctrl_pts_per_side)
        lower_y = np.linspace(0.7, 1.0 - margin_y, num=num_ctrl_pts_per_side)
    elif pattern == 'identity':#水平线
        upper_y = np.linspace(margin_y, margin_y, num=num_ctrl_pts_per_side)
        lower_y = np.linspace(1.0 - margin_y, 1.0 - margin_y, num=num_ctrl_pts_per_side)
    elif pattern == 'sine':#曲线
        upper_y = 0.25 + 0.2 * np.sin(2 * np.pi * upper_x)
        lower_y = 0.75 + 0.2 * np.sin(2 * np.pi * lower_x)
    else:
        raise ValueError('Unknown initialization pattern: {}'.format(pattern))

    init_ctrl_pts = np.concatenate([
        np.stack([upper_x, upper_y], axis=1),
        np.stack([lower_x, lower_y], axis=1),
    ], axis=0)
    print( init_ctrl_pts )
    if activation == 'sigmoid':
        init_biases = -np.log(1. / init_ctrl_pts - 1.+1e-6)
    elif activation == 'none' or activation == None:
        init_biases = init_ctrl_pts
    else:
        raise ValueError('Unknown activation type: {}'.format(activation))

    return init_biases





    




