"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
Created on Thu Mar  7 14:55:44 2019
"""


import glob
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import face_recognition
from code_05_model import *

if __name__ == '__main__':
    # 加载模型
    N = Net()
    N.load_model()
    model = N.model

    # 项目的最终成功：对图片
    classesnum = ['black', 'white']
    num = [0, 0]
    photo = cv2.imread('4.jpg')  # 读取图片数据
    face = face_recognition.face_locations(photo)  # 获取脸部信息
    font = cv2.FONT_ITALIC  # 设置显示字体：斜体
    for (top, right, bottom, left) in face:
        face_image = photo[top:bottom, left:right]
        img = Image.fromarray(face_image)  # 载入到内存中
        img = img.resize((32, 32), Image.ANTIALIAS)  # 图片缩放
        img = np.reshape(img, (-1, 32, 32, 3))  # 加一个维度
        pred = model.predict(img)  # 预测
        print(pred)
        # 返回最大概率值
        classes = pred[0].tolist().index(max(pred[0]))
        # 累计类别数量
        num[classes] = num[classes] + 1
        # 绘制人脸方框
        cv2.rectangle(photo, (left, top), (right, bottom), (0, 0, 255 * classes), 2)
        # 添加类别信息文字
        cv2.putText(photo, str(classesnum[classes]), (left, top - 5), font, 0.5, (0, 0, 255 * classes), 1)
    # 添加计数文字
    cv2.putText(photo, 'black:' + str(num[0]), (10, 30), font, 0.5, (0, 0, 0), 2)
    cv2.putText(photo, 'white:' + str(num[1]), (10, 50), font, 0.5, (0, 0, 255), 2)
    cv2.imshow("photo", photo)
    cv2.waitKey(0)


