"""
Created on Wed Sep 11 11:08:36 2019
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com 
"""
import cv2
import numpy as np
import os
import pandas as pd
from tensorflow.python.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

model_weights_file = './traincpkmodel/1/weights.h5'  # 模型权重文件

image_dimension = 224  # 图片大小

class_names = 'Atelectasis,Cardiomegaly,Consolidation,Edema,Effusion,\
Emphysema,Fibrosis,Hernia,Infiltration,Mass,Nodule,Pleural_Thickening,\
Pneumonia,Pneumothorax'.split(",")  # 类别名称

# 创建模型
base_DenseNet121_model = DenseNet121(include_top=False, weights=None, pooling="avg")  # 创建模型
m_output = base_DenseNet121_model.output
predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(m_output)  # 创建一个全连接层，预测类别
final_conv_layer = base_DenseNet121_model.get_layer("bn")  # 获取模型的最后一个卷积层
model = Model(inputs=base_DenseNet121_model.input, outputs=[predictions, final_conv_layer.output],
              name='myDenseNet121')  # 一个输入，两个输出
if os.path.exists(model_weights_file) == False:
    print("____wrong!!!___no model___:", model_weights_file)
    raise ("wrong")
# 加载模型权重
model.load_weights(model_weights_file)  # 加载模型权重
class_weights = model.layers[-1].get_weights()[0]  # 最后一个网络层的权重


def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


# 在测试影像图上绘制病兆热力图
def plotCMD(photoname, output_file, predictions, conv_outputs):
    img_ori = cv2.imread(photoname)  # 读取原始测试图片

    if img_ori is None:
        raise ("no file!")
        return
    cam = np.reshape(conv_outputs, (-1, 1024))  # conv_outputs.shape = 1，7，7，1024
    class_weights_w = np.reshape(class_weights[:, predictions], (class_weights.shape[0], 1))  # (1024, 1)
    cam = cam @ class_weights_w
    cam = np.reshape(cam, (7, 7))  # 矩阵变成7*7大小
    cam /= np.max(cam)  # 归一化到[0 1]
    cam = cv2.resize(cam, (img_ori.shape[1], img_ori.shape[0]))  # 特征图变到原始图片大小
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # 绘制热力图
    heatmap[np.where(cam < 0.2)] = 0  # 病兆热力图阈值0.2
    img = heatmap * 0.5 + img_ori  # 在原影像图上叠加病兆热力图
    cv2.imwrite(output_file, img)  # 保存图片
    return


from PIL import Image

output_dir = './traincpkmodel/1/cam/'  # 输出病兆图片目录

os.makedirs(output_dir, exist_ok=True)  # os.mkdir


def show_cam(data_dir, file_name):  # 用CAM方法在原影像图上显示病兆热力图

    print(f"process image: {file_name}")
    # #带标签的多个子图
    imageone = Image.open(os.path.join(data_dir, file_name)).resize((image_dimension, image_dimension))  # 调整图片大小
    image_array = np.asarray(imageone.convert("RGB"))  # 变成RGB格式
    image_array = preprocess_input(np.expand_dims(image_array, axis=0))  # 在前面扩展1维，变成[1 …]形式
    img_transformed = image_array  # np.reshape(image_array,[image_dimension,image_dimension,3])

    logits, final_conv_layer = model(img_transformed)  # 使用前面创建的模型
    #    print(final_conv_layer,logits)
    predictions = list(filter(lambda x: logits[0][x] > 0.45, range(0, len(logits[0]))))  # 阈值是0.45
    if len(predictions) == 0:
        return
    print(predictions)
    for i in predictions[:3]:
        output_file = os.path.join(output_dir, f"{i}_{class_names[i]}.{file_name}")
        print('output:', output_file)
        plotCMD(os.path.join(data_dir, file_name), output_file, i, final_conv_layer)  # 调用绘制病兆热力图函数


# 测试影像图的标注文件路径
test_dataset_csv_file = os.path.join(r'./data/default_split', "test.csv")
test_data_dir = r'./data/images'  # 测试影像图

# 读取测试影像图的标注内容
df_images = pd.read_csv(test_dataset_csv_file, header=None, skiprows=1)
# 测试影像图的Image Index
col = df_images.iloc[:, 0]
# 取表中的第0列的所有值
filenames = col.values
# 输出结果
for name in filenames:
    a = show_cam(test_data_dir, name)  # 调用病兆绘图函数
