import pandas as pd
import numpy as np
from tensorflow import keras
import os

save_path = '../model'

# 加载Fer2013数据集
df = pd.read_csv('../数据集/fer2013.csv')

# 准备图像数据和标签
X_train = []
y_train = []
X_test = []
y_test = []

for index, row in df.iterrows():
    pixels = [int(pixel) for pixel in row['pixels'].split(' ')]
    if row['Usage'] == 'PrivateTest':
        X_test.append(np.array(pixels).reshape(48, 48))
        y_test.append(row['emotion'])
    elif row['Usage'] == 'Training':
        X_train.append(np.array(pixels).reshape(48, 48))     # reshape(48, 48)将图像数据表示为二维矩阵的形式，以便于卷积神经网络等模型进行图像特征提取、分类等任务
        y_train.append(row['emotion'])


# 将图像数据缩放到0-1之间并转换为float类型
X_train = np.array(X_train) / 255.0
X_test = np.array(X_test) / 255.0

# 将标签转换为one-hot编码格式
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 定义CNN模型
# Sequential是Keras中的一种模型类型，用于快速构建串行的神经网络模型。
# 用户只需要按照顺序添加每一层的参数即可，Keras会自动将这些层组合成一个模型。
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(7, activation='softmax')
])
# 该模型由多个层组成，包括两个卷积层、两个池化层、两个 Dropout 层和两个全连接层。下面对每个层的作用进行简要说明：
#
# 卷积层：对输入图像执行卷积运算，将其转换为一组特征映射，提取输入图像的局部特征。
# Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1))：
# # 第一个卷积层，使用 32 个 3x3 的卷积核对输入图像进行卷积运算，padding='same' 表示在边缘处进行填充以保持输出尺寸不变，
# # activation='relu' 表示使用 ReLU 激活函数激活特征映射，input_shape=(48, 48, 1) 表示输入图像的尺寸为 48x48，通道数为 1（灰度图像）。
# Conv2D(64, (3, 3), padding='same', activation='relu')：第二个卷积层，与第一个卷积层类似，使用 64 个 3x3 的卷积核对输入特征映射进行卷积运算。
# 池化层：对特征映射进行下采样操作，减小特征图尺寸，降低模型复杂度。
# MaxPooling2D((2, 2))：采用最大池化方式，从输入特征映射中提取每个 2x2 区域的最大值，将输出尺寸缩小一半。
# Dropout 层：对输入进行随机丢弃操作，减少过拟合风险。
# Dropout(0.25)：第一个 Dropout 层，以 0.25 的概率随机丢弃输入。
# Dropout(0.5)：第二个 Dropout 层，以 0.5 的概率随机丢弃输入。
# 全连接层：对前面层的输出进行线性变换和非线性变换，将其转换为最终的分类结果。
# Flatten()：将前面层的输出展平为一维向量，作为全连接层的输入。
# Dense(256, activation='relu')：第一个全连接层，包括 256 个神经元，使用 ReLU 激活函数。
# Dense(7, activation='softmax')：第二个全连接层，包括 7 个神经元，使用 softmax 函数将输出转换为概率值，表示输入图像属于每个类别的概率。


# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
# reshape((-1, 48, 48, 1))就是将原始的图像数据集转换为了CNN模型所需的四维张量形状(N, 48, 48, 1)
model.fit(X_train.reshape((-1, 48, 48, 1)), y_train, epochs=50, batch_size=64)


# 评估模型
test_loss, test_acc = model.evaluate(X_test.reshape((-1, 48, 48, 1)), y_test)
print('Test accuracy:', test_acc)



if not os.path.exists(save_path):
    os.mkdir(save_path)

model.save(os.path.join(save_path, 'my_cnn_model.h5'))

#准确率33.91%