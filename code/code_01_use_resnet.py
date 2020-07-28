# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import base64

tf.compat.v1.disable_v2_behavior()  # 以静态图的方式运行
# 输入是string类型
input_imgs = tf.compat.v1.placeholder(shape=None, dtype=tf.string)
# 把base64字符串图像解码成jpeg格式
decoded = tf.image.decode_jpeg(tf.compat.v1.decode_base64(input_imgs), channels=3)
# 用最近邻法调整图像大小到[224,224],因为ResNet50需要输入图像大小是[224,224]
decoded = tf.compat.v1.image.resize_images(decoded, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# 在0位置增加一个值是1的维度，使其成为一个图像
tensorimg = tf.expand_dims(tf.cast(decoded, dtype=tf.float32), 0)

tensorimg = preprocess_input(tensorimg)  # 图像预处理

with tf.compat.v1.Session() as sess:  # 构建一个会话
    sess.run(tf.compat.v1.global_variables_initializer())
    # 加载ResNet50模型
    Reslayer = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    logits = Reslayer(tensorimg)  # 获取模型的输出节点
    # 得到该图片的每个类别的概率值
    prediction = tf.squeeze(tf.cast(tf.argmax(logits, 1), dtype=tf.int32), [0])

    img_path = './dog.jpg'  # 定义测试图片路径

    with open(img_path, "rb") as image_file:
        # 把图像编码成base64字符串格式
        encoded_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")
    img, logitsv, Pred = sess.run([decoded, logits, prediction], feed_dict={input_imgs: encoded_string})
    print('Pred label ID ', Pred)  # 预测的标签ID
    # 从预测结果中取出前3名
    Pred = decode_predictions(logitsv, top=3)[0]
    print('Predicted:', Pred, len(logitsv[0]))

    # 可视化处理，创建一个1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)  # 设置第一个轴是ax1
    ax1.imshow(img)  # 第一个子图显示原始要预测的图片

    # 设置第二个子图为预测的结果，按概率取前3名
    barlist = ax2.bar(range(3), [i[2] for i in Pred])
    barlist[0].set_color('g')  # 颜色设置为绿色

    # 预测结果前3名的柱状图
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    # 竖直显示Top3的标签
    plt.xticks(range(3), [i[1][:15] for i in Pred], rotation='vertical')

    fig.subplots_adjust(bottom=0.2)  # 调整第二个子图的位置
    plt.show()  # 显示图像

    ##########################################保存模型

    save_path = './model'  # 设置模型保存路径
    # 创建用于保存模型的builder对象
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(save_path + 'imgclass')

    # 定义输入签名，X为输入张量
    inputs = {'input_x': tf.compat.v1.saved_model.utils.build_tensor_info(input_imgs)}
    # 定义输出签名， z为最终需要的输出结果张量
    outputs = {'output': tf.compat.v1.saved_model.utils.build_tensor_info(prediction)}
    signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # 将节点的定义和值加到builder中，同时还加入了标签， 还可以使用TRAINING、GPU或自定义
    builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
                                         {'aianaconda_signature': signature})
    builder.save()  # 保存模型
