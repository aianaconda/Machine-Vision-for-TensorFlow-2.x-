"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""
# 引入基础
import os
import pandas as pd  # 需要通过pip install pandas来进行安装
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import tensorflow as tf

# 引入Keras库
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as kb

# 创建图像生成器，用于训练场景的数据增强
datagen = image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=preprocess_input)
# 创建图像生成器，用于测试场景
test_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input)

# 定义数据集的基本参数
batch_size = 12  # 批次大小
image_dimension = 224  # 统一变形的形状大小

data_dir = r'./data/images'  # 定义图片目录
label_dir = r'./data/default_split'  # 定义标签目录
dataset_csv_file = os.path.join(label_dir, "train.csv")  # 定义训练标签
test_dataset_csv_file = os.path.join(label_dir, "test.csv")  # 定义测试标签
test_data_dir = data_dir

# class_names = "Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia".split(",")

# 肺不张：0，心肌肥大：1，实变：2，水肿：3，
# “积液”：4，“肺气肿”：5，“纤维化”：6，“疝气”：7，浸润：8，
# “肿块”：9，“未发现”：10，“结节”：11，“胸膜增厚”：12，
# “肺炎”：13，“气胸”：14

# 读取训练数据文件为dataframe
dataset_df = pd.read_csv(dataset_csv_file)
# 读取测试文件dataframe
test_dataset_df = pd.read_csv(test_dataset_csv_file)


def func(one):
    if one == 'No Finding':
        return []  # No Finding表示没有疾病，变成空列表，表示没有疾病
    return one.split("|")  # 多种疾病，用竖线|分隔成数组


# 定义训练数据标签列
dataset_df['trueLabels'] = dataset_df['Finding Labels'].apply(func)
# 定义测试数据标签列
test_dataset_df['trueLabels'] = test_dataset_df['Finding Labels'].apply(func)
print(dataset_df['trueLabels'][:8])  # 打印前8个标签

# 制作训练数据集
train_generator = datagen.flow_from_dataframe(
    dataframe=dataset_df,
    directory=data_dir,
    x_col="Image Index",  # 图片ID
    y_col='trueLabels',  # 标签列名
    has_ext=True,  # cvs中的文件名是否带扩展名
    target_size=(image_dimension, image_dimension),
    batch_size=batch_size)

#        classes = class_names,  #加入之后，只能加载该类的样本，No finding全部加载不了
#         class_mode='binary',

# 制作测试数据集
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=test_dataset_df,
    directory=test_data_dir,
    x_col="Image Index",  # 图片ID
    y_col='trueLabels',  # 标签列名
    shuffle=False,
    has_ext=True,  # cvs中的文件名是否带扩展名
    target_size=(image_dimension, image_dimension),
    batch_size=1)

print("测试数据集中的class：", validation_generator.classes[:8])

# 当路径错误时，不会报错。只会为空
print(train_generator.filenames[:2])  # 打印两个图片名称

# 输出对样本分析后的结果
print(f"共{train_generator.n}条样本，共分为{len(train_generator.class_indices)}类，\
      分别是{train_generator.class_indices}，批次为{train_generator.batch_size}。")

# 显示数据
for X_batch, y_batch in train_generator:
    print(y_batch[:2])  # 打印两个标签，One-hot编码表示
    for i in range(0, 6):
        plt.subplot(330 + 1 + i)  # 绘制子图
        plt.axis('off')  # 不显示坐标轴
        plt.imshow(X_batch[i].reshape(224, 224, 3))
    plt.show()
    break

# 统计每类病例中，正向样本的个数
countclass = Counter(np.concatenate(train_generator.classes))  # 类别总个数
for oneclass in sorted(train_generator.class_indices.keys()):
    countkey = train_generator.class_indices[oneclass]
    print(oneclass, countclass[countkey])  # 病兆名称及其样本数量


# 只有互斥才会考虑均衡。1-14个类彼此不互斥。所以没有均衡问题。
def get_single_class_weight(pos_counts, total_counts):
    denominator = (total_counts - pos_counts) + pos_counts
    return {
        0: pos_counts / denominator,  # 计算正样本比例
        1: (denominator - pos_counts) / denominator,  # 计算负样本比例
    }


class_weights = [get_single_class_weight(countclass[key],
                                         train_generator.n) for key in sorted(countclass.keys())]

# 打印每个分类的正负样本比例
for oneclass in sorted(train_generator.class_indices.keys()):
    print(oneclass, class_weights[train_generator.class_indices[oneclass]])

#####################模型部分################################
output_dir = "./traincpkmodel/1"  # 保存训练模型输出目录
output_weights_name = 'weights.h5'  # 保存模型文件名
epochs = 20  # 训练次数
output_weights_file = os.path.join(output_dir, output_weights_name)  # 拼接模型路径
print(f"**输出的模型文件为: {output_weights_file} **")

# 构建输出目录
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)  # 创建模型输出目录


# 定义函数，构建模型
def get_model(class_names, model_name="DenseNet121",
              train_weights_path=None, input_shape=None, weights="imagenet"):
    if train_weights_path is not None:  # 不加载预训练模型权重
        weights = None

    # 加载预训练模型，原有1000个神经元的全连接层替换为有14个神经元的全连接层
    base_DenseNet121_model = DenseNet121(
        include_top=False,  # 不加载最后一层
        weights=weights,
        pooling="avg")  # 平均池化
    x = base_DenseNet121_model.output
    predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)  # 14个神经元的全连接层
    model = Model(inputs=base_DenseNet121_model.input, outputs=predictions, name='my' + model_name)  # 函数式API构建模型

    # 再加载训练的模型
    if train_weights_path is not None:
        print(f"load model weights_path: {train_weights_path}")
        model.load_weights(train_weights_path)
    return model


# 定义模型路径
model_weights_file = os.path.join(output_dir, output_weights_name)  # 输出模型路径
if os.path.exists(model_weights_file) == False:
    model_weights_file = None

print("** 加载模型:", model_weights_file)
model_train = get_model(
    train_generator.class_indices.keys(),
    weights='densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',  # 预训练模型
    train_weights_path=model_weights_file,  # 本地模型
    input_shape=(image_dimension, image_dimension, 3))  # 输入图片大小


def myacc(y_true, y_pred):  # 实现基于样本的准确率评估函数
    threshold = 0.5  # 分类阈值
    y_pred = tf.cast(y_pred >= threshold, y_pred.dtype)  # 布尔类型转float类型
    ret = kb.mean(tf.equal(y_true, y_pred), axis=-1)  # 每行统计，如果全正确，则为1，否则小于1
    return kb.mean(tf.equal(ret, 1))  # 统计当前批次正确率


# 实现基于单个分类的准确率评估函数
def funfactory(index):
    def get_one_metric(y_true, y_pred):
        threshold = tf.cast(0.5, y_pred.dtype)  # 分类阈值
        y_pred = tf.cast(y_pred > threshold, y_pred.dtype)  # 如果预测值大于分类阈值，转换成1，否则转换成0
        # 沿着index列获取切片
        indexarray = tf.ones([batch_size], dtype=tf.int32) * index  # 列表，batch_size个元素，值全都是index
        # 堆叠成二维列表[[0 index] [1 index]......[batch_size-1index]]的形式
        indices = tf.stack([tf.range(batch_size), indexarray], axis=1)
        #        print(indices.numpy())                	#输出切片索引：[[0 1] [1 3] [2 2]]
        it = tf.gather_nd(y_true, indices)  # 根据切片索引获得分数
        ip = tf.gather_nd(y_pred, indices)  # 根据切片索引获得分数
        score = kb.mean(tf.equal(it, ip), axis=-1)
        return score

    return get_one_metric  # 返回一个评估函数


def get_metrics():  # 所有评估函数
    funlist = []  # 列表，保存函数用
    for i in range(len(train_generator.class_indices)):
        onefun = funfactory(i)  # 调用自定义的基于单个分类的准确率评估函数
        onefun.__name__ = onefun.__name__ + str(i)  # 添加函数名称
        funlist.append(onefun)  # 追加到列表
    funlist.append('accuracy')  # 添加准确率评估函数
    funlist.append(myacc)  # 添加自定义的评估函数
    return funlist


funlist = get_metrics()  # 调用评估函数
print(funlist)

###############################


initial_learning_rate = 0.01  # 学习率
optimizer = Adam(lr=initial_learning_rate)

model_train.compile(optimizer=optimizer, loss="binary_crossentropy",
                    metrics=funlist)  # 多个二分类




########################回调
checkpoint = ModelCheckpoint(
    output_weights_file,  # 输出模型文件
    # monitor = 'val_loss',  默认的监测值就是val_loss，可以直接注释掉
    save_weights_only=True,  # 只保存权重
    save_best_only=True,  # 只保存在验证集上性能最好的模型
    verbose=1,  # 用进度条显示
    period=2)  # 每2个epoch保存一次文件

patience_reduce_lr = 1  # 触发退化学习判断次数
min_lr = 2e-6  # 最小学习率
callbacks = [  # 回调函数列表
    checkpoint,  # 检查点回调函数
    ReduceLROnPlateau(monitor='val_loss',  # 监测对象
                      factor=0.5,  # 学习率减少因子
                      patience=patience_reduce_lr,  # 触发退化学习判断次数
                      verbose=1,  # 进度条显示
                      mode="auto",  # 判定退化学习率的条件
                      min_lr=min_lr), ]  # 最小学习率

print("** 开始训练 **")

H = model_train.fit_generator(
    generator=train_generator,  # 训练数据集生成器
    epochs=epochs,  # 训练次数
    validation_data=validation_generator,  # 测试数据集生成器
    callbacks=callbacks,  # 回调函数列表
    class_weight=class_weights  #类别比例
)

plt.style.use("ggplot")  # 绘图样式
plt.figure()
N = epochs
# 显示训练数据集损失
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# 显示验证数据集的损失
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# 训练数据集准确率
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# 验证数据的准确率
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")  # 图的标题
plt.xlabel("Epoch #")  # x轴标签
plt.ylabel("Loss/Accuracy")  # y轴标签
plt.legend(loc="upper left")  # 添加图例
plt.savefig("plot.jpg")  # 保存图片
plt.show()  # 显示图形
