import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from skimage.feature import hog

# 读取fer2013.csv文件
data = pd.read_csv('../数据集/fer2013.csv')

# 数据预处理
pixels = data['pixels'].tolist()
X = []
for pixel_sequence in pixels:
    pixel_array = [int(pixel) for pixel in pixel_sequence.split(' ')]
    X.append(pixel_array)
X = np.array(X) / 255.0
y = data['emotion'].values

# 对图像进行特征提取（使用HOG特征）
features = []
for image in X:
    hog_features = hog(image.reshape((48, 48)), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2))
    features.append(hog_features)
X = np.array(features)
y = data['emotion'].values

# 将数据集划分为训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义一个决策树分类器
clf = DecisionTreeClassifier(random_state=42)

# 定义需要调整的参数范围的分布或列表形式
param_dist = {
    'max_depth': [None] + list(randint(1, 20).rvs(5)),
    'min_samples_split': list(randint(2, 20).rvs(5)),
    'min_samples_leaf': list(randint(1, 8).rvs(5)),
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

# 通过RandomizedSearchCV函数进行超参数优化
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42)
print("开始进行超参数优化...")
print("待测试的参数组合有：")
print(param_dist)
print()
random_search.fit(X_train, y_train)

# 输出最优参数
print("超参数优化完成。")
print("最优的参数组合为：", random_search.best_params_)
print()

# 使用训练集和验证集进行模型训练
clf = DecisionTreeClassifier(**random_search.best_params_, random_state=42)
print("使用训练集和验证集进行模型训练...")
clf.fit(X_train, y_train)
print("模型训练完成。")
print()

# 对测试集数据进行预测并计算模型的准确率
y_test_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print("决策树模型在测试集上的准确率为：", accuracy)
