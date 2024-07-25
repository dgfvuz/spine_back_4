import cv2
from PIL import Image
import numpy as np

def read_image(image_path):
    data = Image.open(image_path)
    data_array = np.array(data)
    data = cv2.cvtColor(data_array,cv2.COLOR_RGB2BGR)
    return data

def split_image(image,k=512):
    shape = image.shape
    x1 = 0
    x2 = shape[1]
    if x2 % 8 != 0:
        x1 += (x2 % 8)//2
        x2 -= (x2-x1) % 8
    y1 = k
    y2 = k + x2-x1
    if y2 > shape[0]:
        y1 = shape[0] - (x2-x1)
        y2 = shape[0]
    return image[y1:y2,x1:x2,:],[x1,x2,y1,y2]