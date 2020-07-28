"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""
# 载入框架模块
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
# 载入其它模块
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
import string
import re
import os
# 载入本工程的代码模块
from code_20_CRNNModel import CRNN, ctc_lambda_func
from code_19_mydataset import get_dataset, decodeindex


# 定义函数，返回Callbacks对象
def get_callbacks(output_subdir):
    # 自定义Callbacks方法
    class MyCustomCallback(ModelCheckpoint):
        # 重载on_epoch_end方法，修改Loss和Val_loss的计算方式
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['loss'] = np.mean(logs.get('loss'))
            logs['val_loss'] = np.mean(logs.get('val_loss'))
            super().on_epoch_end(epoch, logs)

    # 自定义Callbacks方法
    checkpoint = MyCustomCallback(

        # 输出模型文件
        output_subdir + '/weights.{epoch:02d}-{loss:.4f}.hdf5',
        monitor='loss',  # 默认的监测值就是val_loss， #在2.0下loss是多个值，会报错,所以需要重载一下
        save_weights_only=True,  # 只保存权重
        save_best_only=True,  # 只保存在验证集上性能最好的模型
        verbose=1,  # 用进度条显示
        period=10)  # 每迭代训练10次，保存一次文件

    return [checkpoint]


# 获得已训练的模型文件
def getbestmodelfile(output_dir):
    # 将文件夹中的模型文件，并按照从大到小排序
    dirlist = sorted(os.listdir(output_dir), reverse=True)
    #    print(dirlist)
    if len(dirlist) == 0:  # 如果没有模型文件，则返回None
        return None
    # 取出模型文件中的浮点型数值（loss值），并按照该值进行从小到大排序
    dirlist.sort(key=lambda x: float(re.findall(r'-?\d+\.?\d*e?-?\d*?', x)[1]))
    # 将loss值最小的文件名返回
    return os.path.join(output_dir, dirlist[0])


batchsize = 64  # 定义批次大小
tagsize = [64, 128]  # 定义数据集的输出样本尺寸 h,w
ch = 1
label_len = 16  # 定义数据集中，每个标签的序列长度
output_dir = 'resultCRNN'  # 定义模型的输出路径

# 定义数据集的配置文件
dataset_config = {
    'batchsize': batchsize,  # 指定批次
    'image_path': r'dataimgcrop/images/',  # 指定图片文件路径
    'gt_path': r'dataimgcrop/gts/',  # 指定标注文件路径
    'tagsize': tagsize,  # 设置输出图片的高和宽
    'ch': ch,
    'label_len': label_len,  # 设置标签序列的总长度
    'characters': '0123456789' + string.ascii_letters,  # 用于将标签转化为索引的字符集
    'charoffset': 3,  # 定义索引中的预留值：0=保留索引  1=pad 2=unk
}
image_dataset, steps_per_epoch = get_dataset(dataset_config, shuffle=False)

# 定义模型的配置文件
model_config = {
    'batchsize': batchsize,  # 指定批次
    'tagsize': tagsize,  # 输出的图片尺度 h,w
    'ch': ch,
    'label_len': label_len,  # 设置标签序列的总长度
    'outputdim': len(dataset_config['characters']) + dataset_config['charoffset'] + 1
}

# 定义模型
CRNN_model = CRNN(model_config)
CRNN_model.summary()  # 输出模型信息

# 加载模型
os.makedirs(output_dir, exist_ok=True)
callbacks = get_callbacks(output_dir)
output_File = getbestmodelfile(output_dir)
if output_File is not None:
    print('load weight for file ：', output_File)
    CRNN_model.load_weights(output_File)

# 定义计算CTCLoss的偏函数
myctc = partial(ctc_lambda_func, model_config=model_config)
# 定义优化器
optimizer = Adam(lr=0.001, amsgrad=True)


# 定义输入标签的格式
input_labels = Input(name='labels', shape=[model_config['label_len']], dtype='float32')

CRNN_model.compile(loss=myctc, optimizer=optimizer, target_tensors=[input_labels])

# 训练模型，迭代400次
CRNN_model.fit(image_dataset, steps_per_epoch=steps_per_epoch, epochs=400, verbose=1,
              validation_data=image_dataset,
              validation_steps=steps_per_epoch, callbacks=callbacks)


# 定义CTC解码运算模型
input_pred = Input(CRNN_model.output.shape[1:])
ctc_decode = K.ctc_decode(input_pred, input_length=tf.ones(K.shape(input_pred)[0]) * input_pred.shape[1])
decode = K.function([input_pred], [ctc_decode[0][0]])  

# 使用模型进行预测
for i, j in image_dataset.take(1):  # 取出一个批次的数据
    y_pred = CRNN_model.predict(i, steps=1)  # 将数据送入模型中进行预测
    #        print(y_pred)
    print(np.shape(y_pred))
    shape = y_pred.shape


    # 将输出的32个序列值解码，生成指定长度的序列
    out = decode([y_pred])[0]
    print(out.shape)  # (64, 17) 解码后长度会多出一个 -1结束符
    print(j.numpy()[0], out[0])  # 输出标签序列和预测序列


    print('loss:', np.mean(myctc(tf.convert_to_tensor(j), tf.convert_to_tensor(y_pred))))
    print(out)

    out = out - dataset_config['charoffset']
    j = j - dataset_config['charoffset']
    for n, outone in enumerate(out):
        # 将归一化的数据转化成像素
        arraydata = np.uint8(np.squeeze(i[n].numpy() * 255))
        #            print(arraydata, np.shape(arraydata))
        # 将模型输出的索引转化成字符串
        result_str = decodeindex(dataset_config['characters'], outone)
        # 将标签索引转化成字符串
        lab_str = decodeindex(dataset_config['characters'], j[n])
        #            print(result_str)
        plt.title('real: %s\npred:%s' % (lab_str, result_str))
        plt.imshow(arraydata, cmap='gray')  # 显示
        plt.show()
        break
