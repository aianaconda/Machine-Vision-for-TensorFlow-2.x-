"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""
import tensorflow as tf
import code_11_mydataset as mydataset
import code_13_MyCenterNet as MyCenterNet

import os
from distutils.version import LooseVersion

if tf.executing_eagerly():  # 判断动态图是否打开，如果打开，则关闭动态图
    tf.compat.v1.disable_v2_behavior()  # 关闭动态图

# 将图清空
if LooseVersion(tf.__version__) >= LooseVersion("1.13"):
    tf.compat.v1.reset_default_graph()
else:
    tf.reset_default_graph()

tagsize = [384, 384]  # 定义数据集的输出尺寸
batch_size = 4
dataset_config = {  # 定义数据集配置参数
    'batchsize': batch_size,
    'image_path': r'data/train/images/',
    'gt_path': r'data/train/ground_truth/',
    'tagsize': tagsize
}
# 定义数据集
train_gen = mydataset.gen_dataset(dataset_config, shuffle=True)

# 定义模型配置参数
model_config = {
    'mode': 'train',  # 定义训练模式，支持'train'和 'test'
    'input_size': tagsize,
    'num_classes': 2,  # 定义分类个数
    'batch_size': batch_size,
    'score_threshold': 0.1,  # 用于模型预测的参数，决定分类结果的分数阀值
    'top_k_results_output': 100,  # 用于模型预测的参数，需要处理关键点的最大个数
}

# 实例化CenterNet模型类
centernet = MyCenterNet.MyCenterNet(model_config, train_gen)
# 载入已有的模型权重文件
save_path = r'./06centernetmodel/loss'
start = centernet.load_weight(os.path.dirname(save_path))

epochs = 600
lr = 0.0004  # 初始的学习率

reduce_lr_epoch = [40, 80]  # 手动实现退化学习率的调节参数

loss = 1000
for i in range(start, epochs):
    print('-' * 25, 'epoch', i, '-' * 25)
    if i in reduce_lr_epoch:  # 手动实现退化学习率
        lr = lr / 2.
        print('reduce lr, lr=', lr, 'now')
    # 用数据集中的全部数据进行依次模型训练
    mean_loss, mean_kloss, mean_sizeloss, mean_offloss, mean_tloss = centernet.train_one_epoch(lr)
    # 输出模型训练过程中的中间信息
    print('>> mean loss', mean_loss, '。 mean_kloss', mean_kloss,
          '。 mean_sizeloss', mean_sizeloss,
          '。 mean_offloss', mean_offloss)
    if loss > mean_loss:  # 保存loss最低的模型权重
        loss = mean_loss
        centernet.save_weight(save_path + str(loss), i)  # 保存训练过程中的模型
    else:
        print('skip this epoch,minloss:', loss)

# 保存训练好的模型
centernet.save_K_weight('./kerasmodel.h5')
centernet.close_session()  # 训练结束，关闭会话
