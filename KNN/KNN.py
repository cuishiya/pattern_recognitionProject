import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# 读取CSV文件
df = pd.read_csv('../数据集/fer2013.csv')

# 获取表情图像像素值列，并转换为numpy数组格式
X = np.array([np.asarray(row.split(' '), dtype=np.uint8) for row in df['pixels']])

# 获取表情标签列，并转换为numpy数组格式
y = df['emotion'].values

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 定义一个决策树分类器
clf = DecisionTreeClassifier(random_state=42)

# 使用训练集进行模型训练
clf.fit(X_train, y_train)

# 对测试集数据进行预测
y_pred = clf.predict(X_test)

# 计算模型的准确率
accuracy = np.mean(y_pred == y_test)

print("决策树模型的准确率为：", accuracy)
