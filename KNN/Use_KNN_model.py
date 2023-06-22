from joblib import load
from sklearn.metrics import accuracy_score

# 从本地文件中加载模型
knn = load('my_knn_model.joblib')

# 加载数据集
# X_new_raw = df[df['Usage']=='PublicTest']['pixels'].apply(lambda x: np.array(x.split(), dtype=int))
# y_new = np.array(df[df['Usage']=='PublicTest']['emotion'])

# 提取 HOG 特征
X_new = np.array([hog_feat(x.reshape(48, 48)) for x in X_new_raw])

# 对新数据进行预测
y_pred_new = knn.predict(X_new)

# 计算准确率
accuracy_new = accuracy_score(y_new, y_pred_new)
print('KNN模型在新数据集上的准确率为：{:.2f}%'.format(accuracy_new*100))

