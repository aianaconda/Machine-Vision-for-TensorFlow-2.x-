# coding:utf-8
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""
import os
import cv2
import imageio
import face_recognition
import time
from tqdm import tqdm

sampleNum = 0


# 对黑人和白人文件夹中的图片，截取人脸图片
def readFilePath(sample_dir, save_dir):
    # 获取每一张图片
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        for filename in tqdm(filenames):
            # 获取脸部信息并把脸部图片储存
            writeFaceJPG(dirpath, filename, save_dir)


def writeFaceJPG(filename_path, photo_name, save_dir):
    # 图片计数器
    global sampleNum
    img = cv2.imread(os.path.join(filename_path, photo_name))
    if img is None:
        return

    photo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 获取脸部信息
    faces = face_recognition.face_locations(photo)
    for (top, right, bottom, left) in faces:
        sampleNum = sampleNum + 1
        # 判断是否已经存在文件夹
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存图片
        imageio.imwrite(save_dir + "/" + str(sampleNum)
                        + "_" + str(round(time.time())) + ".jpg", photo[top:bottom, left:right])


if __name__ == '__main__':
    readFilePath(sample_dir='org_black/', save_dir='./dataset/black')
    readFilePath(sample_dir='org_white/', save_dir='./dataset/white')
