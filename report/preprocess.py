from PIL import Image
import os
import cv2
import numpy as np
from memory_profiler import profile

RESULT_PATH = "./model_runtime/mid_result/"

# 输入图片，返回处理后的图片
def resize(img):
    return img.resize((512, 1280), Image.LANCZOS)


def cut_image(img):
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row, col = image.shape

    col_histogram = []
    row_histogram = []
    col_target = 200
    for i in range(row):
        count = np.int64(0)
        for j in range(col):
            count = count + image[i][j]

        row_histogram.append(count)

    row_index = row_histogram.index(min(row_histogram))

    while row_index > row / 2:
        row_histogram.remove(min(row_histogram))
        row_index = row_histogram.index(min(row_histogram))

    leg_row = find_leg(image)
    for i in range(col):
        count = np.int64(0)
        for j in range(row_index, leg_row):
            count = count + image[j][i]
        # print(count)

        col_histogram.append(count)
    col_index = col_histogram.index(max(col_histogram))
    while col_index < col / 4 or col_index > 3 * col / 4:
        col_histogram.remove(max(col_histogram))
        col_index = col_histogram.index(max(col_histogram))
    crop = np.ones(shape=(leg_row - row_index, 2 * col_target),dtype=np.uint8)
    for i in range(leg_row - row_index):
        for j in range(2 * col_target):
            if col_index - col_target + j < 0 or col_index - col_target + j >= col: continue
            crop[i][j] = image[row_index + i][col_index - col_target + j]
    return crop,[[row,col],[row_index,leg_row,col_index - col_target,col_index + col_target]]


def change_histogram(img):
    '''
    第一步：使用 cv2.createCLAHE(clipLimit=2.0, titleGridSize=(8, 8)) 实例化均衡直方图函数
    第二步：使用 .apply 进行均衡化操作
    第三步：进行画图操作
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    '''
    clipLimit：颜色对比度的阈值
    titleGridSize：进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
    '''
    cl1 = clahe.apply(img)
    return cl1


def toSquare(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w, _ = image.shape

    result_image = np.zeros((h, h, 3), dtype=np.uint8)

    left = int((h - w) / 2)
    right = int((h - w) / 2 + w)

    for i in range(result_image.shape[0]):
        for j in range(result_image.shape[1]):
            if j < left or j >= right:
                continue
            for k in range(result_image.shape[2]):
                result_image[i, j, k] = image[i, j - left, k]

    result_image = cv2.resize(result_image, (512, 512))
    return result_image, [h,left, right]


def find_leg(image):
    row, col = image.shape
    row_array = []
    for i in range(int(row / 2), row):
        find_leg1 = 0
        find_black = 0
        find_leg2 = 0
        find_black2 = 0
        wrong = 0
        for j in range(col):
            if find_leg1 == 0 and image[i][j] > 0:
                find_leg1 = j
                continue
            if find_black == 0 and find_leg1 > 0 and image[i][j] == 0:
                find_black = j
                if find_black - find_leg1 < 150:
                    find_leg1 = 0
                    find_black = 0
                continue
            if find_leg2 == 0 and find_black > 0 and image[i][j] > 0:
                find_leg2 = j
                continue
            if find_black2 == 0 and ((find_leg2 > 0 and image[i][j] == 0) or j == col - 1):
                find_black2 = j
                if find_black2 - find_leg2 < 150:
                    find_black2 = 0
                    find_leg2 = 0
                continue
            # if find_black2 > 0 and image[i][j] > 0:
            #     wrong = 1
            #     break
            # if find_black2 > 0 and (find_black-find_leg1 < 40 or find_black2-find_leg2 <40):
            #     wrong = 1
            #     break
        # print(i,find_leg1,find_black,find_leg2,find_black2,wrong)
        if find_leg1 > 0 and find_black > 0 and find_leg2 > 0 and find_black2 > 0 and wrong == 0:
            # row_array.append(i)
            return i - 100

    # print("file,row:", row_array)
    return row

# cut_detail = [[row,col],[row_index,leg_row,col_index - col_target,col_index + col_target]]
# toSquare_detail = [h,left, right]
def preprocess_image(origin_path):
    image = Image.open(origin_path)
    width = image.size[0]
    height = image.size[1]
    resized_img = resize(image)
    cuted_img,cut_detail = cut_image(resized_img)
    histogram_img = change_histogram(cuted_img)
    processed_img, toSquare_detail = toSquare(histogram_img)
    # cv2转为PIL
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_img = Image.fromarray(processed_img)
    processed_img.save(RESULT_PATH + origin_path.split('/')[-1])
    return RESULT_PATH + origin_path.split('/')[-1],[[width,height],cut_detail,toSquare_detail]

# 输入的result是cv2的图片
def reverse_process(result, preprocess_detail):
    # toSquare_detail = [h,left, right]
    # cut_detail = [[row,col],[row_index,leg_row,col_index - col_target,col_index + col_target]]
    # preprocess_detail = [[width,height],cut_detail,toSquare_detail]
    width = preprocess_detail[0][0]
    height = preprocess_detail[0][1]
    h = preprocess_detail[2][0]
    left = preprocess_detail[2][1]
    right = preprocess_detail[2][2]
    row = preprocess_detail[1][0][0]
    col = preprocess_detail[1][0][1]
    row_index = preprocess_detail[1][1][0]
    leg_row = preprocess_detail[1][1][1]
    col_index_sub_col_target = preprocess_detail[1][1][2]
    col_index_add_col_target = preprocess_detail[1][1][3]
    result = cv2.resize(result, (h, h), interpolation=cv2.INTER_CUBIC)
    result = result[:, left:right]
    result_to_origin = np.zeros((row, col, 3), dtype=np.uint8)
    result_to_origin[row_index:leg_row, col_index_sub_col_target:col_index_add_col_target] = result
    origin_recovered = cv2.resize(result_to_origin, (width, height), interpolation=cv2.INTER_CUBIC)
    return origin_recovered

# preprocess_detail = [[width,height],cut_detail,toSquare_detail]
# point = [x,y]
# 返回原图坐标
def point_to_origin(point,preprocess_detail):
    print("point:",point)
    width = preprocess_detail[0][0]
    height = preprocess_detail[0][1]
    h = preprocess_detail[2][0]
    left = preprocess_detail[2][1]
    row = preprocess_detail[1][0][0]
    col = preprocess_detail[1][0][1]
    row_index = preprocess_detail[1][1][0]
    col_index_sub_col_target = preprocess_detail[1][1][2]
    x = point[0]
    y = point[1]
    x = int(x * h / 512)
    y = int(y * h / 512)
    x = x - left
    x = x + col_index_sub_col_target
    y = y + row_index
    x = x*width/col
    y = y*height/row
    x = int(x)
    y = int(y)
    return [x,y]


def distance_to_origin(distance,preprocess_detail):
    width = preprocess_detail[0][0]
    height = preprocess_detail[0][1]
    row = preprocess_detail[1][0][0]
    col = preprocess_detail[1][0][1]
    distance = distance * width / col
    return distance


if __name__ == "__main__":
    path, preprocess_detail = preprocess_image("./X_ray/向克捷_1.JPG")
    print(path)
    image = Image.open(path)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    res = eval(path)
    # 转为cv2
    res = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
    recovered = reverse_process(res, preprocess_detail)
    # 保存图片
    cv2.imwrite("recovered.jpg", recovered)
