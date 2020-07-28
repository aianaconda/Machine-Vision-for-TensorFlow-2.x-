# -*- coding: utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""

import json
import base64
import requests

# 客户端对读取的图像进行编码，到服务端需要再解码
with open("dog.jpg", "rb") as image_file:
    # 读取图片，用urlsafebase编码，然后转换为str
    encoded_string = str(base64.urlsafe_b64encode(image_file.read()), "utf-8")

# 编码转换为json格式
data = json.dumps({"inputs": encoded_string, "signature_name": "aianaconda_signature"})
print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

# 定义请求的内容格式
headers = {"content-type": "application/json"}
# 向服务器发送请求
json_response = requests.post('http://localhost:8501/v1/models/md:predict', data=data, headers=headers)
# 打印请求返回的内容，即预测结果
print(json_response.text)

