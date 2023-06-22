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

print('数据采集开始')
for index, row in df.iterrows():
    pixels = [int(pixel) for pixel in row['pixels'].split(' ')]
    if row['Usage'] == 'PrivateTest':
        X_test.append(np.array(pixels).reshape(48, 48))
        y_test.append(row['emotion'])
    elif row['Usage'] == 'Training':
        X_train.append(np.array(pixels).reshape(48, 48))
        y_train.append(row['emotion'])

print('数据采集结束')

# 将图像数据缩放到0-1之间并转换为float类型
X_train = np.array(X_train, dtype=np.float32) / 255.0
X_test = np.array(X_test, dtype=np.float32) / 255.0

# 将标签转换为one-hot编码格式
y_train = keras.utils.to_categorical(y_train, num_classes=7)
y_test = keras.utils.to_categorical(y_test, num_classes=7)

# 定义CNN模型
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(rate=0.25),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(rate=0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(units=7, activation='softmax')
])

# 编译模型
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print('编译结束')

# 定义回调函数
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(save_path, 'my_cnn_parm.h5'), save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# 训练模型
print('开始训练')
model.fit(X_train.reshape(-1, 48, 48, 1), y_train, epochs=25, batch_size=16,
      validation_data=(X_test.reshape(-1, 48, 48, 1), y_test), verbose=2,
      callbacks=[checkpoint_cb, early_stopping_cb])

print('训练结束')
# 评估模型
test_loss, test_acc = model.evaluate(X_test.reshape((-1, 48, 48, 1)), y_test)
print('Test accuracy:', test_acc)

# 保存模型
if not os.path.exists(save_path):
    os.mkdir(save_path)
model.save(os.path.join(save_path, 'my_cnn_model_v2.h5'))

# 代码优化方面主要包括：
#
# 使用BatchNormalization层：批量归一化可以使得网络更稳定，加快模型收敛速度，提高模型表现。
#
# 增加卷积核数量和卷积层数：增加每个卷积层中的卷积核数量和增加卷积层数，有可能更好地拟合数据集，提高模型准确率。
#
# 增加正则化项：增加BatchNormalization层以及使用Dropout层来防止过拟合。
#
# 调整学习率：调整Adam优化器的学习率为0.001。
#
# 引入回调函数：使用ModelCheckpoint和EarlyStopping回调函数分别保存最优的模型参数和在验证集上提前终止训练，避免训练过程中出现过拟合和梯度消失等问题。





