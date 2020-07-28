"""
Created on Wed Sep 11 11:08:36 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
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

    classesnum = ['black','white']
    cap = cv2.VideoCapture(0)
    while(True):
        # 获取摄像头信息
        ret, frame = cap.read()
        photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 图片色彩处理
        face = face_recognition.face_locations(photo) # 获取脸部信息
        font = cv2.FONT_ITALIC # 斜体
        num = [0,0] # 初始化计数
        for (top, right, bottom, left) in face:
            face_image = photo[top:bottom, left:right]
            img = Image.fromarray(face_image) # 载入到内存中
            img = img.resize((32, 32), Image.ANTIALIAS) # 图片缩放
            img = np.reshape(img, (-1,32, 32, 3)) # 加一个维度
            pred = model.predict(img) # 预测
            # 返回最大概率值
            classes = pred[0].tolist().index(max(pred[0]))
            # 累计类别数量
            num[classes] = num[classes] + 1
            cv2.rectangle(photo, (left, top), (right, bottom), (0, 0, 255*classes), 2)
            # 添加类别信息文字
            cv2.putText(photo,str(classesnum[classes]),(left,top-5), font, 0.5,(0, 0, 255*classes), 1)
        # 添加计数文字
        cv2.putText(photo,'black:'+str(num[0]),(10,30), font, 0.5,(255, 255, 255), 2)
        cv2.putText(photo,'white:'+str(num[1]),(10,50), font, 0.5,(0, 0, 255), 2)
        cv2.imshow("photo", photo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 最后要释放捕获
    cap.release()
    cv2.destroyAllWindows()
