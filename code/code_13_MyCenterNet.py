"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""
import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras.layers import *
from code_12_hourglass import *
from distutils.version import LooseVersion

isNewAPI = True  # 获得当前版本
if LooseVersion(tf.__version__) >= LooseVersion("1.13"):
    print("new version API")
else:
    print('old version API')
    isNewAPI = False
x = tf.constant(1, shape=[2, 3])


# 定义Center Net模型类
class MyCenterNet:
    def __init__(self, config, dataset_gen):  # 初始化
        self.mode = config['mode']  # 运行方式
        self.input_size = config['input_size']  # 输入尺寸
        self.num_classes = config['num_classes']  # 分类个数
        self.batch_size = config['batch_size']  # 测试使用
        if dataset_gen != None:  # 初始化数据集张量
            self.train_initializer, self.train_iterator = dataset_gen
        self.score_threshold = config['score_threshold']  # 预测结果的阀值
        self.top_k_results_output = config['top_k_results_output']  # 处理预测结果的最大个数

        if isNewAPI == True:  # 定义迭代训练的计步张量
            self.global_step = tf.compat.v1.get_variable(name='global_step', initializer=tf.constant(0),
                                                         trainable=False)
        else:
            self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)

        self._define_inputs()  # 定义输入节点
        self._build_graph()  # 搭建网络模型
        # 定义保持模型的张量
        if isNewAPI == True:
            self.saver = tf.compat.v1.train.Saver()
        else:
            self.saver = tf.train.Saver()
        self._init_session()  # 初始化会话

    def _init_session(self):  # 初始化会话
        if isNewAPI == True:
            self.sess = tf.compat.v1.InteractiveSession()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = tf.InteractiveSession()
            self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.sess.run(self.train_initializer)

    def close_session(self):  # 关闭会话
        self.sess.close()

    def _define_inputs(self):  # 定义输入节点

        std = tf.constant([0.229, 0.224, 0.225])  # 展开，还可以tf.convert_to_tensor将变量转换成张量
        mean = tf.constant([0.485, 0.456, 0.406])

        mean = tf.reshape(mean, [1, 1, 1, 3])
        std = tf.reshape(std, [1, 1, 1, 3])

        if isNewAPI == True:
            placeholder = tf.compat.v1.placeholder
            decode_base64 = tf.io.decode_base64
            decode_jpeg = tf.io.decode_jpeg
        else:
            placeholder = tf.placeholder
            decode_base64 = tf.decode_base64
            decode_jpeg = tf.image.decode_jpeg

        if self.mode == 'train':  # 训练和测试，使用不同的输入节点
            self.imgs, self.ground_truth = self.train_iterator.get_next()  # 从数据集获取样本和标签
            self.imgs.set_shape([self.batch_size, self.input_size[0], self.input_size[1], 3])
            self.images = (self.imgs / 255. - mean) / std  # 对样本进行归一化
            self.lr = placeholder(dtype=tf.float32, shape=[], name='lr')  # 定义学习率输入节点
        else:
            self.imgs = placeholder(shape=None, dtype=tf.string)  # 以base64的方式获取图片数据
            decoded = decode_jpeg(decode_base64(self.imgs), channels=3)
            image = tf.expand_dims(decoded, 0)
            if isNewAPI == True:
                image = tf.image.resize(image, self.input_size, tf.image.ResizeMethod.BILINEAR)
            else:
                image = tf.image.resize_images(image, self.input_size, tf.image.ResizeMethod.BILINEAR)

            self.images = (image / 255. - mean) / std  # 对图片进行归一化
            # 定义标签占位符（支持测试功能）
            self.ground_truth = placeholder(tf.float32, [self.batch_size, None, 5], name='labels')

    def _build_graph(self):  # 定义函数，搭建网络
        kwargs = {  # 配置沙漏模型
            'num_stacks': 2,
            'cnv_dim': 256,
            'inres': self.input_size,
        }
        heads = {  # 配置输出节点
            'hm': self.num_classes,  # 分类
            'reg': 2,  # size
            'wh': 2  # offset
        }
        # 生成沙漏模型
        self.model = HourglassNetwork(heads=heads, **kwargs)
        # 根据运行方式，设置模型是否可以训练
        if self.mode == 'train':
            self.model.trainable = True
        else:
            self.model.trainable = False
        # 将输入节点传入模型
        outputs = self.model(self.images)
        # 取第二个沙漏的输出节点
        keypoints, size, offset = outputs[3], outputs[4], outputs[5]
        keypoints = tf.nn.sigmoid(keypoints)  # 对关键点做sigmoid变换

        pshape = [tf.shape(offset)[1], tf.shape(offset)[2]]  # 获得输出特征图的尺寸[96,96]

        h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
        w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
        [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)  # 生成网格,两个变量的形状均为[96,96]

        stride = 4.0  # 定义缩小比例
        if self.mode == 'train':  # 训练模式下，计算loss
            total_loss = []
            kloss, sizeloss, offloss = [], [], []

            for i in range(self.batch_size):
                loss = self._compute_one_image_loss(keypoints[i, ...], offset[i, ...], size[i, ...],
                                                    self.ground_truth[i, ...], meshgrid_y, meshgrid_x,
                                                    stride, pshape)
                kloss.append(loss[0])
                sizeloss.append(loss[1])
                offloss.append(loss[2])
                total_loss.append(loss[0] + loss[1] + loss[2])
            # 收集loss值，用于显示
            self.lossinfo = [tf.reduce_mean(kloss), tf.reduce_mean(sizeloss), tf.reduce_mean(offloss),
                             tf.reduce_mean(total_loss)]
            # 对3个loss取平均值，作为总的loss
            self.loss = tf.reduce_mean(total_loss)



            if isNewAPI == True:
                optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
            else:
                optimizer = tf.train.AdamOptimizer(self.lr)

            print('optimizer:', optimizer)

            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        else:  # 预测模式下，计算预测结果
            meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)
            meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
            center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)  # 生成网格
            # 获取每个网格的分类索引
            category = tf.expand_dims(tf.squeeze(tf.argmax(keypoints, axis=-1, output_type=tf.int32)), axis=-1)
            meshgrid_xyz = tf.concat([tf.zeros_like(category), tf.cast(center, tf.int32), category], axis=-1)
            # 根据索引获取每个网格的分类分数
            keypoints = tf.gather_nd(keypoints, meshgrid_xyz)
            keypoints = tf.expand_dims(keypoints, axis=0)
            keypoints = tf.expand_dims(keypoints, axis=-1)
            # 将网格中，3*3区域内的最大分类分数取出，其它分数变为0
            keypoints_peak = MaxPool2D(pool_size=3, strides=1, padding='same')(keypoints)
            keypoints_mask = tf.cast(tf.equal(keypoints, keypoints_peak), tf.float32)
            keypoints = keypoints * keypoints_mask

            scores = tf.reshape(keypoints, [-1])  # 每个网格的分类分数
            class_id = tf.reshape(category, [-1])  # 每个网格的分类类别索引
            bbox_yx = tf.reshape(center + offset, [-1, 2])  # 每个网格的中心点偏移
            bbox_hw = tf.reshape(size, [-1, 2])  # 每个网格的预测物体尺寸

            score_mask = scores > self.score_threshold
            scores = tf.boolean_mask(scores, score_mask)  # 对分类分数按照指定阀值过滤
            class_id = tf.boolean_mask(class_id, score_mask)  # 对分类类别索引按照指定阀值过滤
            bbox_yx = tf.boolean_mask(bbox_yx, score_mask)  # 对心点偏移引按照指定阀值过滤
            bbox_hw = tf.boolean_mask(bbox_hw, score_mask)  # 对预测物体尺寸按照指定阀值过滤

            # 根据中心点和尺寸算出边框4个点的坐标
            bbox = tf.concat([bbox_yx - bbox_hw / 2., bbox_yx + bbox_hw / 2.], axis=-1) * stride
            # 计算获取预测结果的个数
            num_select = tf.cond(tf.shape(scores)[0] > self.top_k_results_output, lambda: self.top_k_results_output,
                                 lambda: tf.shape(scores)[0])
            # 按照指定个数获取预测结果
            select_scores, select_indices = tf.nn.top_k(scores, num_select)
            select_class_id = tf.gather(class_id, select_indices)
            select_bbox = tf.gather(bbox, select_indices)

            # 对预测结果进行NMS处理
            selected_indices = tf.image.non_max_suppression(select_bbox, select_scores, self.top_k_results_output)
            # 根据NMS输出的索引，获取最终结果
            selected_boxes = tf.gather(select_bbox, selected_indices)
            selected_scores = tf.gather(select_scores, selected_indices)
            selected_class_id = tf.gather(select_class_id, selected_indices)
            self.detection_pred = [selected_boxes, selected_scores, selected_class_id]

    def _compute_one_image_loss(self, keypoints, offset, size, ground_truth, meshgrid_y, meshgrid_x,
                                stride, pshape):
        # 把填充（-1）前面的ground_truth取出来
        slice_index = tf.argmin(ground_truth, axis=0)[0]  # ground_truth中最小的索引，例如[[2,3],[1,4],[0,4]]，结果为[2, 0]
        ground_truth = tf.gather(ground_truth, tf.range(0, slice_index, dtype=tf.int64))
        # 将标签坐标按照相应的比例缩小
        ngbbox_y = ground_truth[..., 0] / stride
        ngbbox_x = ground_truth[..., 1] / stride
        ngbbox_h = ground_truth[..., 2] / stride
        ngbbox_w = ground_truth[..., 3] / stride
        class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)  # 取出标签中的分类

        ngbbox_yx = ground_truth[..., 0:2] / stride
        ngbbox_yx_round = tf.floor(ngbbox_yx)  # 向下取整
        offset_gt = ngbbox_yx - ngbbox_yx_round  # 获得中心点在缩小后的偏移

        size_gt = ground_truth[..., 2:4] / stride
        ngbbox_yx_round_int = tf.cast(ngbbox_yx_round, tf.int64)  # 中心点的整数值
        # 计算关键点损失
        keypoints_loss = self._keypoints_loss(keypoints, ngbbox_yx_round_int, ngbbox_y, ngbbox_x, ngbbox_h,
                                              ngbbox_w, class_id, meshgrid_y, meshgrid_x, pshape)

        offset = tf.gather_nd(offset, ngbbox_yx_round_int)
        size = tf.gather_nd(size, ngbbox_yx_round_int)
        offset_loss = tf.reduce_mean(tf.abs(offset_gt - offset))
        size_loss = tf.reduce_mean(tf.abs(size_gt - size))

        total_loss = [keypoints_loss, 0.1 * size_loss, offset_loss]
        return total_loss

    def _keypoints_loss(self, keypoints, gbbox_yx, gbbox_y, gbbox_x, gbbox_h, gbbox_w,
                        classid, meshgrid_y, meshgrid_x, pshape):
        sigma = self._gaussian_radius(gbbox_h, gbbox_w, 0.7)  # 计算高斯半径

        # 将标签坐标扩充一个维度
        gbbox_y = tf.reshape(gbbox_y, [-1, 1, 1])
        gbbox_x = tf.reshape(gbbox_x, [-1, 1, 1])
        sigma = tf.reshape(sigma, [-1, 1, 1])

        num_g = tf.shape(gbbox_y)[0]  # 当前图片对应的标签个数
        meshgrid_y = tf.expand_dims(meshgrid_y, 0)
        meshgrid_y = tf.tile(meshgrid_y, [num_g, 1, 1])
        meshgrid_x = tf.expand_dims(meshgrid_x, 0)
        meshgrid_x = tf.tile(meshgrid_x, [num_g, 1, 1])
        # 用高斯核函数计算出每个像素点对应中心点的映射值，生成结果的形状为[num_g,96,96]
        keyp_penalty = tf.exp(-((gbbox_y - meshgrid_y) ** 2 + (gbbox_x - meshgrid_x) ** 2) / (2 * sigma ** 2))
        zero_like_keyp = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32), axis=-1)  # 生成形状为[96,96,1]的0

        reduction = []  #
        gt_keypoints = []  # 用于计算交叉熵时，区分正负loss
        for i in range(self.num_classes):
            exist_i = tf.equal(classid, i)  # 如果标签中有当前类别，则将其设为True。否则设为为False
            # 按照exist_i中为True的索引，去keyp_penalty中取值
            reduce_i = tf.boolean_mask(keyp_penalty, exist_i, axis=0)  # 输出形状为[n,96,96]
            reduce_i = tf.cond(
                tf.equal(tf.shape(reduce_i)[0], 0),
                lambda: zero_like_keyp,
                lambda: tf.expand_dims(tf.reduce_max(reduce_i, axis=0), axis=-1)  # 生成形状为[96,96,1]
            )
            reduction.append(reduce_i)

            gbbox_yx_i = tf.boolean_mask(gbbox_yx, exist_i)  # 取出该类对应的中心点

            if isNewAPI == True:
                gt_keypoints_i = tf.cond(  # 如果当前类中有中心点，则交叉熵的标签为1，否则为0
                    tf.equal(tf.shape(gbbox_yx_i)[0], 0),
                    lambda: zero_like_keyp,
                    lambda: tf.expand_dims(tf.sparse.to_dense(
                        tf.sparse.SparseTensor(gbbox_yx_i, tf.ones_like(gbbox_yx_i[..., 0], tf.float32),
                                               dense_shape=pshape), validate_indices=False),
                        axis=-1)
                )
            else:
                gt_keypoints_i = tf.cond(
                    tf.equal(tf.shape(gbbox_yx_i)[0], 0),
                    lambda: zero_like_keyp,
                    lambda: tf.expand_dims(tf.sparse_tensor_to_dense(
                        tf.SparseTensor(gbbox_yx_i, tf.ones_like(gbbox_yx_i[..., 0], tf.float32), dense_shape=pshape),
                        validate_indices=False),
                        axis=-1)
                )
            gt_keypoints.append(gt_keypoints_i)
        reduction = tf.concat(reduction, axis=-1)
        gt_keypoints = tf.concat(gt_keypoints, axis=-1)

        # 计算focal loss
        if isNewAPI == True:
            keypoints_pos_loss = -tf.math.pow(1. - keypoints, 2.) * tf.math.log(keypoints + 1e-12) * gt_keypoints
            keypoints_neg_loss = -tf.math.pow(1. - reduction, 4) * tf.math.pow(keypoints, 2.) * tf.math.log(
                1. - keypoints + 1e-12) * (1. - gt_keypoints)
        else:
            keypoints_pos_loss = -tf.pow(1. - keypoints, 2.) * tf.log(keypoints + 1e-12) * gt_keypoints
            keypoints_neg_loss = -tf.pow(1. - reduction, 4) * tf.pow(keypoints, 2.) * tf.log(
                1. - keypoints + 1e-12) * (1. - gt_keypoints)

        keypoints_loss = tf.reduce_sum(keypoints_pos_loss) / tf.cast(num_g, tf.float32) + tf.reduce_sum(
            keypoints_neg_loss) / tf.cast(num_g, tf.float32)
        return keypoints_loss

    # 计算高斯半径（与h、w相关的标准差）from cornernet
    def _gaussian_radius(self, height, width, min_overlap=0.7):
        a1 = 1.
        b1 = (height + width)
        c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
        sq1 = tf.sqrt(b1 ** 2. - 4. * a1 * c1)
        r1 = (b1 + sq1) / (2.*a1)

        a2 = 4.
        b2 = 2. * (height + width)
        c2 = (1. - min_overlap) * width * height
        sq2 = tf.sqrt(b2 ** 2. - 4. * a2 * c2)
        r2 = (b2 + sq2) / (2.*a2)

        a3 = 4. * min_overlap
        b3 = -2. * min_overlap * (height + width)
        c3 = (min_overlap - 1.) * width * height
        sq3 = tf.sqrt(b3 ** 2. - 4. * a3 * c3)
        r3 = (b3 + sq3) / (2.*a3)

        return tf.reduce_min([r1, r2, r3])

    def train_one_epoch(self, lr):  # 定义方法，训练模型
        self.sess.run(self.train_initializer)
        mean_loss = []
        mean_kloss, mean_sizeloss, mean_offloss, mean_tloss = [], [], [], []

        i = 0
        while True:  # 循环使用数据进行训练
            try:
                _, loss, lossinfo = self.sess.run([self.train_op, self.loss, self.lossinfo], feed_dict={self.lr: lr})
                i = i + 1
                sys.stdout.write('\r>> ' + 'iters ' + str(i) + str(':') + ' loss ' + str(loss) +
                                 ' kloss ' + str(lossinfo[0]) + ' sizeloss ' + str(lossinfo[1]) + ' offloss ' + str(
                    lossinfo[2]))
                sys.stdout.flush()
                mean_loss.append(loss)
                mean_kloss.append(lossinfo[0])
                mean_sizeloss.append(lossinfo[1])
                mean_offloss.append(lossinfo[2])
                mean_tloss.append(lossinfo[3])
            except tf.errors.OutOfRangeError:  # 利用异常出发数据集遍历结束
                # print("遍历结束")
                break # 跳出循环
        sys.stdout.write('\n')
        mean_loss = np.mean(mean_loss)
        mean_kloss = np.mean(mean_kloss)
        mean_sizeloss = np.mean(mean_sizeloss)
        mean_offloss = np.mean(mean_offloss)
        mean_tloss = np.mean(mean_tloss)
        # 返回训练过程中的loss值
        return mean_loss, mean_kloss, mean_sizeloss, mean_offloss, mean_tloss

    def test_one_image(self, images):  # 定义方法，测试模型
        pred = self.sess.run(self.detection_pred, feed_dict={self.imgs: images})
        return pred

    def save_K_weight(self, path):  # 保存Keras格式的检查点文件
        self.model.save_weights(path)

    def load_K_weight(self, path):  # 载入Keras格式的检查点文件
        self.model.load_weights(path)
        return 0

    def save_weight(self, path, epoch):  # 保存TensorFlow格式的检查点文件
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.saver.save(self.sess, path, global_step=epoch)

    def load_weight(self, path):

        if not os.path.exists(path):  # 载入keras格式的模型文件
            print("there is not a model path:", path)
            if os.path.exists('./kerasmodel.h5'):
                print('load kerasmodel.h5')
                self.load_K_weight('./kerasmodel.h5')
            else:
                print('no kerasmodel.h5')

        else:  # 载入TensorFlow格式的模型文件
            if isNewAPI == True:
                kpt = tf.compat.v1.train.latest_checkpoint(path)  # 查找最新的检查点
            else:
                kpt = tf.train.latest_checkpoint(path)  # 查找最新的检查点
            print("load model:", kpt, path)
            if kpt != None:
                self.saver.restore(self.sess, kpt)  # 还原模型
                ind = kpt.find("-")
                print('load weight', kpt, 'successfully')
                return int(kpt[ind + 1:])
        return 0


