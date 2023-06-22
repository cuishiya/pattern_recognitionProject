# -*-coding=utf-8-*-
# AUTHOR：ADLcarus
# TIME：2023-06-20/18:10
# FILENAME：multi_logistic_regression.py
# SOFTWARE：PyCharm


import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_process import data_process_main
import joblib
import datetime

np.set_printoptions(precision=3)


def model():
    # 超参数
    feature_num = 300
    max_iter = 1000

    # 数据处理
    Raw_X_data, Raw_Y_data = data_process_main()

    X_data = Raw_X_data
    Y_data = Raw_Y_data

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

    # 对训练集和测试集进行 LBP 特征提取
    orb = cv2.ORB_create()
    X_train_lbp = []
    X_test_lbp = []
    y_train_lbp = []
    y_test_lbp = []
    for i, img in enumerate(X_train):
        grey_img = np.reshape(img, (128, 128))
        kp, des = orb.detectAndCompute(grey_img, None)
        # 若不存在特征点，则移除该数据
        if kp:
            lbp_feature = np.concatenate([kp[0].pt, des.ravel()])
            if len(lbp_feature) < feature_num:
                del X_train[i]
                del y_train[i]
            else:
                X_train_lbp.append(lbp_feature[:feature_num])
                y_train_lbp.append(y_train[i])

        else:
            del X_train[i]
            del y_train[i]

    for i, img in enumerate(X_test):
        grey_img = np.reshape(img, (128, 128))
        kp, des = orb.detectAndCompute(grey_img, None)
        # 若不存在特征点，则移除该数据
        if kp:
            lbp_feature = np.concatenate([kp[0].pt, des.ravel()])
            if len(lbp_feature) < feature_num:
                del X_test[i]
                del y_test[i]
            else:
                X_test_lbp.append(lbp_feature[:feature_num])
                y_test_lbp.append(y_test[i])
        else:
            del X_test[i]
            del y_test[i]

    print("调整后的训练数据量", "X_train", len(X_train_lbp), "y_train", len(y_train_lbp))
    print("调整后的测试数据量", "X_test", len(X_test_lbp), "y_test", len(y_test_lbp))
    print("调整后的数据量", "X", len(X_train_lbp) + len(X_test_lbp), "Y", len(y_train_lbp) + len(y_test_lbp))

    # 将 LBP 特征点转换为 NumPy 数组
    X_train_lbp = np.asarray(X_train_lbp, dtype=object)
    X_test_lbp = np.asarray(X_test_lbp, dtype=object)

    # 模型训练
    # 定义并训练多元逻辑回归模型
    multi_logistic_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=max_iter)
    multi_logistic_model.fit(X_train_lbp, y_train_lbp)

    # # 加载测试数据并进行预测
    # # 在测试集上进行预测并计算准确率
    y_pred = multi_logistic_model.predict(X_test_lbp)
    accuracy = np.sum(y_pred == y_test_lbp) / len(y_test_lbp)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # 打印预测结果
    print(y_pred[:20])
    print(y_test_lbp[:20])

    # 保存模型
    model_name = f"../model/model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{accuracy}.pkl"
    joblib.dump(multi_logistic_model, model_name)


if __name__ == "__main__":
    model()
