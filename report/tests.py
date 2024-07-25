from django.test import TestCase
from PIL import Image
import cv2
import numpy as np
from preprocess import preprocess_image, reverse_process
from model import eval

# Create your tests here.
if __name__ == "__main__":
    path, preprocess_detail = preprocess_image("./X_ray/丁子航_psFEXF1.jpg")
    print(path)
    image = Image.open(path)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    res = eval(path)
        # 转为cv2
    res = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
    recovered = reverse_process(res, preprocess_detail)
    # 读取图片
    image = Image.open("./X_ray/丁子航_psFEXF1.jpg")
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # 将recovered中的白色部分设置为image中的对应部分
    # recovered = cv2.bitwise_and(image, recovered)
    # 保存图片
    print(recovered.shape)
    
    cv2.imwrite("recovered.jpg", recovered)