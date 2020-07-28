# -*- coding: utf-8 -*-

"""
Created on Wed Sep 11 11:08:36 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
@改错实例
@运行版本：tf1.13.1  tf2
"""

from captcha.image import ImageCaptcha  #pip captcha
import matplotlib.pyplot as plt
import numpy as np
import random

import string
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters)

generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

plt.imshow(img)
plt.title(random_str)

############################
def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)  
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = np.array(generator.generate_image(random_str)) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

X, y = next(gen(1))
plt.imshow(X[0])
plt.title(decode(y))


from tensorflow.keras.models import *
from tensorflow.keras.layers import *

input_tensor = Input((height, width, 3))
x = input_tensor
for i in range(4):
    x = Conv2D(32*2**i, 3, 1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32*2**i, 3, 1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)


x = Flatten()(x)
x = Dropout(rate=0.25)(x)
out = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]


model = Model(inputs=input_tensor, outputs=out)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])



model.fit_generator(gen(), steps_per_epoch=500, epochs=50, 
                    validation_data=gen(), validation_steps=10)

X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')



