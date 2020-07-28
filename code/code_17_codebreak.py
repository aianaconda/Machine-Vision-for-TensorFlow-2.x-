# -*- coding: utf-8 -*-

"""
Created on Wed Sep 11 11:08:36 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""


from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string

characters = string.digits + string.ascii_uppercase
print(characters)  # 将获取的基础字符输出
# 定义验证码的尺寸，字符长度
width, height, n_len, n_class = 210, 80, 6, len(characters)

myfonts = [r'./ttf/a.ttf', r'./ttf/b.ttf', r'./ttf/c.ttf', r'./ttf/d.ttf']
vCodeobj = ImageCaptcha(width=width, height=height, fonts=myfonts)
# 随机生成字符串
random_str = ''.join([random.choice(characters) for j in range(4)])
img = vCodeobj.generate_image(random_str)  # 根据字符串生成验证码

plt.imshow(img)
plt.title(random_str)


# '''

############################
def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height, fonts=myfonts)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            # 生成验证码图片，并将其归一化
            X[i] = np.array(generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


def decode(y):  # 将验证码对应的标签转化为字符
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])


# 测试数据集
X, y = next(gen(1))
plt.imshow(X[0])
plt.title(decode(y))

#
#
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

input_tensor = Input((height, width, 3))
# 搭建卷积模型
x = input_tensor
for i in range(4):
    for iii in range(2):
        # 实现2个valid类型的卷积操作
        x = Conv2D(32 * 2 ** i, (3, 1), activation='relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(32 * 2 ** i, (1, 3), activation='relu')(x)
        x = BatchNormalization()(x)
    # 下采样
    x = Conv2D(32 * 2 ** i, 2, 2, activation='relu')(x)
    x = BatchNormalization()(x)

out = []  # 定义输出列表
for i in range(n_len):
    onecode = Conv2D(n_class, (1, 9))(x)  # 同尺度的卷积就是全连接
    onecode = Reshape((n_class,))(onecode)
    onecode = Activation('softmax', name='c%d' % (i + 1))(onecode)
    out.append(onecode)

model = Model(inputs=input_tensor, outputs=out)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, amsgrad=True),
              metrics=['accuracy'])

model.load_weights(r'./my_model/mymodel.h5')
model.fit_generator(gen(), steps_per_epoch=500, epochs=6,
                   validation_data=gen(), validation_steps=10)

X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s' % (decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')

import os

os.makedirs(r'./my_model', exist_ok=True)  # os.mkdir
model.save_weights(r'./my_model/mymodel.h5')  # 保存模型的权值到外部文件


