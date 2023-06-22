from keras.models import load_model
import numpy as np
from PIL import Image
import os


# 加载模型和定义标签
model_path = '../model/my_model.h5'
model = load_model(model_path)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

image_dir = '../picture/'

def predict_emotion(image):
    # 缩放图像并转换为灰度图像
    gray_image = image.convert('L').resize((48, 48))

    # 转换为三维数组(48, 48, 1)
    gray_image = np.reshape(gray_image, (1, 48, 48, 1))

    # 归一化图像像素值
    gray_image = gray_image / 255.0

    # 预测表情
    prediction = model.predict(gray_image)

    return EMOTIONS[np.argmax(prediction)]


for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        filepath = os.path.join(image_dir, filename)
        image = Image.open(filepath)
        predicted_emotion = predict_emotion(image)
        print(filename, 'predicted emotion:', predicted_emotion)
