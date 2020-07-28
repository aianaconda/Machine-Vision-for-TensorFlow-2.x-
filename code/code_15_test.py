"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
Created on Thu Mar  7 14:55:44 2019
"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image#, ImageDraw
from tensorflow.keras.preprocessing import image
from distutils.version import LooseVersion
import base64

import code_11_mydataset as mydataset
import code_13_MyCenterNet as MyCenterNet

if tf.executing_eagerly():
    tf.compat.v1.disable_v2_behavior() #关闭动态图
    
isNewAPI = True
if LooseVersion(tf.__version__) >= LooseVersion("1.13"):
    print("new version API")
    tf.compat.v1.reset_default_graph() 
else:
    print('old version API')
    isNewAPI = False
    tf.reset_default_graph() 
    
tagsize = [384,384]
batch_size=1

model_config = {
    'mode': 'test',                                       # 'train', 'test'
    'input_size': tagsize,
    'num_classes': 2,#20,
    'batch_size': batch_size,
    #test
    'score_threshold': 0.04,
    'top_k_results_output': 100,                           
}

save_path ='./06centernetmodel/loss'
centernet = MyCenterNet.MyCenterNet(model_config, None)
start = centernet.load_weight(os.path.dirname(save_path))



imgdir = r'data/test/images/194.jpg'
with open(imgdir, "rb") as image_file:
    imgs = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")
img = image.load_img(imgdir, target_size=tagsize,interpolation='bilinear')
img = np.expand_dims(img, 0)

result = centernet.test_one_image(imgs)
#result = centernet.test_one_image(img)

bbox = result[0]
scores = result[1]
class_id = result[2]
print(scores,bbox,class_id)

img = Image.fromarray(np.uint8(  np.squeeze(img) ))
mydataset.showimgwithbox(img,bbox,scores,y_first = True)



