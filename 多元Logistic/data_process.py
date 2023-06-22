# -*-coding=utf-8-*-
# AUTHOR：ADLcarus
# TIME：2023-06-20/1:30
# FILENAME：data_process.py
# SOFTWARE：PyCharm


import os
import sys
import cv2
import numpy as np
import re
import random
import data_augmentation_func  # 导入数据增强的函数


# 数据转换为图片
def data_convert():
    # 设定相关参数
    database_path = r"../数据集/rawdata/"
    processed_data_save_path = r'../数据集/processed_data/'
    file_name_list = os.listdir(database_path)  # 读取数据文件名称
    convert_error_index_list = []
    data_list = []

    # # 读取文件
    num = len(file_name_list)
    for i in range(0, num):

        img_path = database_path + file_name_list[i]
        img_array = np.fromfile(img_path, dtype=np.uint8)

        if len(img_array) != (128 * 128):
            convert_error_index_list.append(file_name_list[i])
        else:
            data_list.append(img_array.copy())
            img = np.reshape(img_array, (128, 128))
            cv2.imwrite(processed_data_save_path + file_name_list[i] + '.png', img)
            # # 采用cv进行显示
            # cv2.namedWindow("test")
            # cv2.imshow("test", img)
            # cv2.waitKey(0)

    print("错误数据序号", convert_error_index_list)

    return convert_error_index_list


# 数据统计和处理
def data_process():
    # 设定相关参数
    database_path = r"../数据集/rawdata/"
    label_dir_path = r'../数据集/label/raw_label_text/'
    label_save_dir_path = r'../数据集/label/'
    processed_data_save_path = r'../数据集/processed_data/'
    file_name_list = os.listdir(database_path)  # 读取数据文件名称

    # 读取标签的文本文件，并逐行存储到列表中
    # 这里只提取样本标签中的表情标签
    label_name_list = []
    label_type_list = ['error']  # 类别
    error_label_index_list = []  # 标签缺失的数据的名称
    label_file_name = os.listdir(label_dir_path)
    for i in range(len(label_file_name)):
        # 路径拼接
        label_file_path = label_dir_path + label_file_name[i]
        # 逐行读取文件
        with open(label_file_path, 'r') as f:
            text = f.readlines()
            for j in range(len(text)):
                data_num = re.search(r"\d+", text[j])
                data_label = re.search(r"_face\s+(\w+)", text[j])  # '\s+'是一个或者多个空白字符，(\w+)为需要返回的文件
                if data_label:
                    label_name_list.append(data_num.group() + "-" + data_label.group(1))
                    if data_label.group(1) not in label_type_list:
                        label_type_list.append(data_label.group(1))
                else:
                    label_name_list.append(data_num.group() + "-" + "error")
                    error_label_index_list.append(data_num.group())

    # 逐行打印
    # for j in range(len(label_name_list)):
    #     print(label_name_list[j])

    # 保存标签
    with open(label_save_dir_path + 'label.txt', 'w') as f:
        for j in range(len(label_name_list)):
            f.write(label_name_list[j] + '\n')

    print("标签数", len(label_name_list))
    print("实际样本数", len(file_name_list))
    print("标签类别为", label_type_list)
    print("标签缺失的数据编号", error_label_index_list)


# 数据编码,返回X和Y标签，其中X为一维度的向量
def data_encoder(error_index_list):
    # 参数
    label_path = r'..\数据集\label\label.txt'

    # 变量
    encoder_label_list = []  # 编码后的Y标签，整数
    data_list = []  # 与Y标签对应的X数据，一维变量
    data_index_list = []  # 数据文件的名称，例如5106

    # 标签和对应的编码
    str_label_list = ['smiling', 'serious', 'funny']
    int_label_list = [0, 1, 2]
    each_category_num_list = [0, 0, 0]
    data_list_for_each_category = [[] for i in range(len(str_label_list))]  # 根据类别存储的标签

    # 读取标签和序号
    # 同时筛选掉错误的标签数据
    with open(label_path, 'r') as f:
        text = f.readlines()
        for i in range(len(text)):
            data_index = re.search(r"\d+", text[i]).group()
            label = re.search(r"-(\w+)", text[i]).group(1)

            if label in str_label_list and data_index not in error_index_list:
                index = [i for i, e in enumerate(str_label_list) if e == label][0]
                data_index_list.append(data_index)
                encoder_label_list.append(int_label_list[index])

                # 统计类别数量
                each_category_num_list[int_label_list[index]] += 1

                # 按类别存储
                data_list_for_each_category[int_label_list[index]].append(data_index)

    # 读取数据
    for i in range(len(data_index_list)):
        path =  path = r"..\\数据集/rawdata\\" + str(data_index_list[i])
        X = np.fromfile(path, dtype=np.uint8)
        data_list.append(np.reshape(X, (1, -1)))

    print("标签", str_label_list, "编码", int_label_list, "类别数量", each_category_num_list)
    print("类别数量二次统计", [len(data_list_for_each_category[0]), len(data_list_for_each_category[1]), len(data_list_for_each_category[2])])

    return data_list, data_index_list, encoder_label_list, data_list_for_each_category, str_label_list, int_label_list


# 数据增强
def data_augmentation(index_list, label, generate_data_num):
    # 设置属性
    path = r"..//数据集/processed_data//"
    img_save_path = r"..//数据集/augmentation_data//img//"
    data_save_path = r"..//数据集/augmentation_data//data//"
    data_list = []

    new_index_list = []

    # 加载彩色图像，并转换为灰度图
    file_index = 6000
    for i in range(generate_data_num):
        # 随机选择一张图片进行增强
        rnd_index = random.randint(0, len(index_list) - 1)
        img_path = path + index_list[rnd_index] + '.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 从写好的数据增强函数中,随机选择一种进行图像增强
        augment_img = data_augmentation_func.random_augment(img)

        # 存储图片
        cv2.imwrite(img_save_path + str(file_index) + '.png', augment_img)

        # # 采用cv进行显示
        # cv2.namedWindow("test")
        # cv2.namedWindow("test2")
        # cv2.imshow("test", img)
        # cv2.imshow("test2", augment_img)
        # cv2.waitKey(0)

        data_list.append(np.reshape(augment_img, (1, -1)))  # 一维化，并且进行存储

        # 存储一维化后的数据
        np.save(data_save_path + str(file_index), np.reshape(augment_img, (1, -1)))

        new_index_list.append(file_index)
        file_index += 1

    new_label_list = [label for i in range(generate_data_num)]

    return data_list, new_index_list, new_label_list


def data_process_main():

    convert_error_index_list = data_convert()  # 数据转换为图片
    data_process()
    old_data_list, data_index_list, encoder_label_list, data_list_for_each_category, str_label_list, int_label_list = data_encoder(
        convert_error_index_list)
    data_list, new_index_list, new_encoder_label_list = data_augmentation(data_list_for_each_category[2], int_label_list[2], 1700)

    # 合并数据集，并且返回X_data和Y_data
    total_index_list = data_index_list + new_index_list
    total_label_list = encoder_label_list + new_encoder_label_list

    # 数据增强后的标签数
    print("数据清洗后的数据量", len(total_index_list))
    print("数据清洗后的标签数", len(total_label_list))

    X_Data = old_data_list + data_list
    Y_Data = encoder_label_list + new_encoder_label_list
    # print(len(X_Data), len(Y_Data))
    # print(X_Data[0:20], Y_Data[0:20])
    # print(total_index_list[0:20])
    # print(total_label_list[0:20])

    return X_Data, Y_Data


if __name__ == "__main__":
    data_process_main()
