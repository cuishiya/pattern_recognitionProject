import cv2
import numpy as np
import random


# 定义不同的图像增强方法
def brightness(image):
    # 随机缩放亮度因子
    factor = random.uniform(0.5, 1.5)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def contrast(image):
    # 随机缩放对比度因子
    factor = random.uniform(0.5, 1.5)
    return cv2.convertScaleAbs(image, alpha=factor, beta=128 * (1 - factor))


def sharpen(image):
    # 使用锐化算子增强图像
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def blur(image):
    # 随机选择模糊程度并使用高斯模糊
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def flip(image):
    # 随机水平或垂直翻转图像
    # axis = random.choice([0, 1])
    # return cv2.flip(image, axis)
    return cv2.flip(image, 1)  # 只进行水平翻转


def scale(image):
    # 随机缩放图像大小
    scale_factor = random.uniform(0.75, 1.25)
    h, w = image.shape[:2]
    new_h, new_w = int(scale_factor * h), int(scale_factor * w)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if scale_factor < 1:
        pad_top = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        pad_bottom = h - new_h - pad_top
        pad_right = w - new_w - pad_left
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return padded
    else:
        crop_top = (new_h - h) // 2
        crop_left = (new_w - w) // 2
        cropped = resized[crop_top:crop_top + h, crop_left:crop_left + w]
        return cropped


# 组合不同的增强方法
def random_augment(image):
    funcs = [brightness, contrast, sharpen, blur, flip, scale]
    random.shuffle(funcs)
    for func in funcs:
        image = func(image)
    return image
