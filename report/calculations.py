import os
import cv2
import numpy as np
import math
from .utils import read_image,split_image
from PIL import Image
import pytesseract
from .Unet import get_Unet_output
import imutils
from .TTS import TTS_eval
from .VR import VR_eval
from memory_profiler import profile
 
# @AUTHOR: yasiare
# @description: 用来找到图像中像素厘米比
# @param: img 是一个cv2的图像
# @return: 厘米像素比(即多少像素表示1cm),如果找不到则返回28
def find_centermeter_per_pixel(image):
    if image.shape[2] != 3:
        print("请输入RGB图像,不要输入灰度图像")
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,100,100])
    upper_yellow = np.array([30,255,255])
    mask_yellow = cv2.inRange(hsv,lower_yellow,upper_yellow)
    image = cv2.bitwise_and(image,image,mask=mask_yellow)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edged = cv2.Canny(image,20,130,apertureSize=3)
    # 膨胀
    kernel = np.ones((3,3),np.uint8)
    edged = cv2.dilate(edged,kernel,iterations=1)
    # 霍夫变换
    lines = cv2.HoughLinesP(edged,1,np.pi/180,1000)
    # 寻找最长的一条边
    if lines is None:
        return 28.5
    result_line = None
    for line in lines:
        if result_line is None:
            result_line = line
        elif abs(line[0][3] - line[0][1]) + abs(line[0][2] - line[0][0]) > abs(result_line[0][3] - result_line[0][1]) + abs(result_line[0][2] - result_line[0][0]):
            result_line = line
    line = result_line[0]
    # 找到y值大的点
    if line[1] > line[3]:
        point = [line[0],line[1]]
    else:
        point = [line[2],line[3]]
    x2 = min(image.shape[1] - 1,point[0] + 100)
    y2 = min(image.shape[0] - 1,point[1] + 200)
    x1 = max(0,point[0] - 150)
    y1 = point[1]
    cut_img = image[y1:y2,x1:x2]
    cut_img = Image.fromarray(cv2.cvtColor(cut_img,cv2.COLOR_BGR2RGB))
    cut_img.save('cut_img.jpg')
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Tesseract-OCR'
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cut_img,config=custom_config)
    text = text.split('\n')[0]
    # 判断字符串是否是纯数字
    if text.isdigit():
        return (abs(line[3] - line[1]) - 5) / int(text)
    else:
        return 28.5

# 用于在二维空间中寻找线段延长线上的点
def yc_line(center, yc_point, dist):
    c_x, c_y = center
    t_x, t_y = yc_point
    x_dst = np.sqrt((center[0] - yc_point[0]) ** 2)
    y_dst = np.sqrt((center[1] - yc_point[1]) ** 2)
    # print(x_dst, y_dst)
    if t_y == c_y:
        if t_x < c_x:
            new_point = (t_x - dist, t_y)
        else:
            new_point = (t_x + dist, t_y)
        return new_point

    if t_x == c_x:
        if t_y < c_y:
            new_point = (t_x, t_y - dist)
        else:
            new_point = (t_x, t_y + dist)
        return new_point

    angle = np.arctan(y_dst / x_dst)
    angle_dg = angle * 180 / np.pi
    x_diff = np.cos(angle) * dist
    y_diff = np.sin(angle) * dist
    # print(x_diff, y_diff)
    if t_x < c_x and t_y < c_y:
        new_point = (t_x - x_diff, t_y - y_diff)
    elif t_x < c_x and t_y > c_y:
        new_point = (t_x - x_diff, t_y + y_diff)
    elif t_x > c_x and t_y < c_y:
        new_point = (t_x + x_diff, t_y - y_diff)
    else:
        new_point = (t_x + x_diff, t_y + y_diff)
    # print(angle, angle_dg)
    # print(new_point)
    new_point = (int(new_point[0]), int(new_point[1]))
    return new_point

# 骶骨（S1）倾斜角
def find_SCRL(image):
    # path = './output/447.jpg'
    # image = cv2.imread(path)
    background = image[:, :, 0]
    # print(background.shape)
    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形

        # print(rotatedRect[2])
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
            # rotate.append(1)
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下

        angle = 180 - math.fabs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
        # print(angle)
        if y1 > y2:
            angle = -angle

        # if y1 > y2:
        #     cv2.line(image, (x1 - 80, y1), (x1 + 100, y1), thickness=2, color=(0, 0, 255))
        #     yc_x, yc_y = yc_line((x1, y1), (x2, y2), 50)
        #     cv2.line(image, (x1, y1), (yc_x, yc_y), thickness=2, color=(0, 0, 255))
        #     cv2.putText(image, str(round(angle, 2)), (yc_x, yc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # else:
        #     cv2.line(image, (x2 - 100, y2), (x2 + 80, y2), thickness=2, color=(0, 0, 255))
        #     yc_x, yc_y = yc_line((x2, y2), (x1, y1), 50)
        #     cv2.line(image, (yc_x, yc_y), (x2, y2), thickness=2, color=(0, 0, 255))
        #     cv2.putText(image, str(round(angle, 2)), (yc_x, yc_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # cv2.imwrite(result_path, image)
        break
    # 将angle转换为基本数据类型
    angle = float(angle)
    # 将坐标转换为基本数据类型
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    return angle,[[x1,y1],[x2,y2]]

# 未测试左高右低的情况 T1（C7下一块）倾斜角
def find_T1_tile_angle(image):
    background = image[:, :, 0]
    # print(background.shape)

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in reversed(contours):
        area = cv2.contourArea(c)
        if area < 150:
            # print("find cobb area:", path, contours_count, perimeter, area)
            continue

        if count < 1:
            count += 1
            continue

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
        # print(rotatedRect[2])
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
            # rotate.append(1)
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下
            # rotate.append(0)

        angle = 180 - math.fabs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
        # print(angle)
        if y1 > y2:
            angle = -angle
        break
    # # 注释
    #     if y1 > y2:
    #         cv2.line(image, (x1 - 50, y1), (x1 + 50, y1), thickness=2, color=(0, 0, 255))
    #         yc_x, yc_y = yc_line((x1, y1), (x2, y2), 20)
    #         cv2.line(image, (x1, y1), (yc_x, yc_y), thickness=2, color=(0, 0, 255))
    #         cv2.putText(image, str(round(angle, 2)), (yc_x, yc_y + 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #     else:
    #         cv2.line(image, (x2 - 50, y2), (x2 + 50, y2), thickness=2, color=(0, 0, 255))
    #         yc_x, yc_y = yc_line((x2, y2), (x1, y1), 20)
    #         cv2.line(image, (yc_x, yc_y), (x2, y2), thickness=2, color=(0, 0, 255))
    #         cv2.putText(image, str(round(angle, 2)), (yc_x, yc_y - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #     break
    # cv2.imwrite("./T1_angle.jpg", image)

    # 将angle转换为基本数据类型
    angle = float(angle)
    # 将坐标转换为基本数据类型
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    return angle,[[x1,y1],[x2,y2]]

# S1中垂线
def find_CSVL(c):
    rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
    box = cv2.boxPoints(rotatedRect)
    box = np.int_(box)

    if rotatedRect[2] > 45:
        x1, y1 = box[0]  # 左上
        x2, y2 = box[1]  # 右上
        x3, y3 = box[2]  # 右下
        x4, y4 = box[3]  # 左下
        # rotate.append(1)
    else:
        x1, y1 = box[1]  # 左上
        x2, y2 = box[2]  # 右上
        x3, y3 = box[3]  # 右下
        x4, y4 = box[0]  # 左下
        # rotate.append(0)

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

def draw_CSVL(image):
    # path = './output/98.jpg'
    background = image[:, :, 0]

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_count = 0
    for c in contours:
        contours_count += 1
        area = cv2.contourArea(c)
        if (area < 150 and contours_count > 8) or (area < 350 and contours_count <= 4) or (
                area < 300 and 4 < contours_count <= 8): continue

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上

        center_x, center_y = find_CSVL(c)

        # cv2.line(image, (x1, y1), (x2, y2), thickness=2, color=(0, 0, 255))
        # # drawdottedline(image, (center_x, center_y), (center_x, center_y + 100),thickness=2, color=(0, 0, 255))
        # cv2.arrowedLine(image, (center_x, center_y + 50), (center_x, center_y - 100), thickness=2, color=(0, 0, 255))
        # # cv2.line(image, (center_x, center_y), (center_x, center_y - 80), thickness=1, color=(0, 0, 255))
        break

    # cv2.imwrite("./csvl.jpg", image)

    # 将坐标转换为基本数据类型
    center_x = int(center_x)
    center_y = int(center_y)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    CSVL = center_x
    return CSVL,[[x1,y1],[x2,y2],[center_x,center_y]]

# C7中垂线
def find_C7PL(contour):
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    return center_x, center_y

def draw_C7PL(image):
    # path = './output/98.jpg'
    background = image[:, :, 0]

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in reversed(contours):
        area = cv2.contourArea(c)
        if area < 150:
            continue

        center_x, center_y = find_C7PL(c)
        # cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)
        # cv2.arrowedLine(image, (center_x, center_y), (center_x, center_y + 100), thickness=2, color=(0, 0, 255))
        break

    # cv2.imwrite("./c7pl.jpg", image)

    # 将坐标转换为基本数据类型
    center_x = int(center_x)
    center_y = int(center_y)
    C7PL = center_x
    return C7PL,[[center_x,center_y]]
    

# image:二值化图像,以胸12为分界,寻找胸弯和腰弯的顶椎
def find_chest_waist_apex(CSVL_x, C7PL_x, image):
    contours, cnt = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    contours_count = -1
    # 遍历锥体, 将每个锥体的四个顶点坐标存入lines
    for c in contours:
        perimeter = cv2.arcLength(c, True)  # 计算这个轮廓的周长
        area = cv2.contourArea(c)
        contours_count += 1
        if area < 150:
            continue

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
        # # print(rotatedRect[2])
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        # # print("box:",box)
        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下

        line_point = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        lines.append(line_point)

    # 找到两个特殊的点，右上角和右下角
    T12_x3, T12_y3 = lines[6][2]
    T12_x4, T12_y4 = lines[6][3]
    if T12_y3 < T12_y4:
        T12_y = T12_y4
    else:
        T12_y = T12_y3

    hight, width = image.shape
    # print(background.shape)
    chest_left_x = 0
    chest_left_y = 0
    chest_right_x = 0
    chest_right_y = 0
    for i in range(width):
        for j in range(T12_y):
            if image[j, i] == 255:
                chest_left_x = i
                chest_left_y = j
                break
        if chest_left_x != 0:
            break

    for i in range(width):
        for j in range(T12_y):
            if image[j, width - 1 - i] == 255:
                chest_right_x = width - 1 - i
                chest_right_y = j
                break
        if chest_right_x != 0:
            break

    waist_left_x = 0
    waist_left_y = 0
    waist_right_x = 0
    waist_right_y = 0
    for i in range(width):
        for j in range(T12_y, hight):
            if image[j, i] == 255:
                waist_left_x = i
                waist_left_y = j
                break
        if waist_left_x != 0:
            break

    for i in range(width):
        for j in range(T12_y, hight):
            if image[j, width - 1 - i] == 255:
                waist_right_x = width - 1 - i
                waist_right_y = j
                break
        if waist_right_x != 0:
            break

    chest_right_distance = math.fabs(C7PL_x - chest_right_x)
    chest_left_distance = math.fabs(C7PL_x - chest_left_x)
    if chest_right_distance > chest_left_distance:
        chest_distance = chest_right_distance
        chest_direction = "right"
    else:
        chest_distance = chest_left_distance
        chest_direction = "left"

    waist_right_distance = math.fabs(CSVL_x - waist_right_x)
    waist_left_distance = math.fabs(CSVL_x - waist_left_x)
    if waist_right_distance > waist_left_distance:
        waist_distance = waist_right_distance
        waist_direction = "right"
    else:
        waist_distance = waist_left_distance
        waist_direction = "left"

    if waist_distance > chest_distance:
        if waist_direction == "right":
            return chest_left_x, chest_left_y, waist_right_x, waist_right_y
        else:
            return chest_right_x, chest_right_y, waist_left_x, waist_left_y
    else:
        if chest_direction == "right":
            return chest_right_x, chest_right_y, waist_left_x, waist_left_y
        else:
            return chest_left_x, chest_left_y, waist_right_x, waist_right_y

def find_apex_rectangle(apex_x, apex_y, width, lines):
    apex_rectangle = 0
    if apex_x < width / 2:
        for line in lines:
            up_x, up_y = line[0]
            down_x, down_y = line[3]
            if up_y < apex_y < down_y:
                apex_rectangle = lines.index(line)
                break
    else:
        for line in lines:
            up_x, up_y = line[1]
            down_x, down_y = line[2]
            if up_y < apex_y < down_y:
                apex_rectangle = lines.index(line)
                break
    return apex_rectangle


# 顶椎偏距,寻找两个顶椎： 胸弯顶椎和腰弯顶椎
def find_avt(image):
    # path = './output/447.jpg'

    background = image[:, :, 0]

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)  # 计算这个轮廓的周长
        area = cv2.contourArea(c)

        if area < 150:
            continue

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形

        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下

        line_point = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        lines.append(line_point)

    CSVL_x, CSVL_y, C7PL_x, C7PL_y = 0, 0, 0, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue

        CSVL_x, CSVL_y = find_CSVL(c)
        break

    for c in reversed(contours):
        area = cv2.contourArea(c)
        if area < 150:
            continue
        C7PL_x, C7PL_y = find_C7PL(c)
        break

    chest_x, chest_y, waist_x, waist_y = find_chest_waist_apex(CSVL_x, C7PL_x, threshold)

# # 注释
#     if math.fabs(chest_x - C7PL_x) > math.fabs(waist_x - CSVL_x):
#         apex_x = chest_x
#         apex_y = chest_y
#     else:
#         apex_x = waist_x
#         apex_y = waist_y

#     if C7PL_x == CSVL_x:
#         avt_distance = apex_x - C7PL_x
#         cv2.line(image, (apex_x, apex_y - 80), (apex_x, apex_y + 80), thickness=2, color=(0, 0, 255))
#         if avt_distance > 0:
#             cv2.arrowedLine(image, (CSVL_x, apex_y), (CSVL_x + (avt_distance - 5), apex_y), thickness=2,
#                             color=(0, 0, 255), tipLength=0.3)
#             cv2.putText(image, str(avt_distance), (apex_x + 5, apex_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#         else:
#             cv2.arrowedLine(image, (CSVL_x, apex_y), (CSVL_x + (avt_distance + 5), apex_y), thickness=2,
#                             color=(0, 0, 255), tipLength=0.3)
#             cv2.putText(image, str(avt_distance), (apex_x - 40, apex_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         apex_rectangle = find_apex_rectangle(apex_x, apex_y, width, lines)

#         # print(apex_rectangle)
#         box = cv2.boxPoints(cv2.minAreaRect(contours[apex_rectangle]))
#         box = np.int0(box)
#         cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
#         cv2.line(image, (CSVL_x, CSVL_y + 100), (C7PL_x, C7PL_y - 100), thickness=2, color=(0, 0, 255))
#     else:
#         avt_chest_distance = chest_x - C7PL_x
#         avt_waist_distance = waist_x - CSVL_x
#         cv2.line(image, (chest_x, chest_y - 80), (chest_x, chest_y + 80), thickness=2, color=(0, 255, 0))
#         if avt_chest_distance > 0:
#             cv2.arrowedLine(image, (C7PL_x, chest_y), (C7PL_x + (avt_chest_distance - 5), chest_y), thickness=2,
#                             color=(0, 255, 0), tipLength=0.3)
#             cv2.putText(image, str(avt_chest_distance), (chest_x + 5, chest_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 255, 0), 2)
#         else:
#             cv2.arrowedLine(image, (C7PL_x, chest_y), (C7PL_x + (avt_chest_distance + 5), chest_y), thickness=2,
#                             color=(0, 255, 0), tipLength=0.3)
#             cv2.putText(image, str(avt_chest_distance), (chest_x - 40, chest_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 255, 0), 2)

#         cv2.line(image, (waist_x, waist_y - 80), (waist_x, waist_y + 80), thickness=2, color=(0, 0, 255))
#         if avt_waist_distance > 0:
#             cv2.arrowedLine(image, (CSVL_x, waist_y), (CSVL_x + (avt_waist_distance - 5), waist_y), thickness=2,
#                             color=(0, 0, 255), tipLength=0.3)
#             cv2.putText(image, str(avt_waist_distance), (waist_x + 5, waist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 0, 255), 2)
#         else:
#             cv2.arrowedLine(image, (CSVL_x, waist_y), (CSVL_x + (avt_waist_distance + 5), waist_y), thickness=2,
#                             color=(0, 0, 255), tipLength=0.3)
#             cv2.putText(image, str(avt_waist_distance), (waist_x - 40, waist_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 0, 255), 2)

#         waist_rectangle = find_apex_rectangle(waist_x, waist_y, width, lines)
#         chest_rectangle = find_apex_rectangle(chest_x, chest_y, width, lines)

#         chest_box = cv2.boxPoints(cv2.minAreaRect(contours[chest_rectangle]))
#         chest_box = np.int_(chest_box)
#         cv2.drawContours(image, [chest_box], 0, (0, 255, 0), 2)

#         waist_box = cv2.boxPoints(cv2.minAreaRect(contours[waist_rectangle]))
#         waist_box = np.int_(waist_box)
#         cv2.drawContours(image, [waist_box], 0, (0, 0, 255), 2)
#         cv2.arrowedLine(image, (C7PL_x, C7PL_y), (C7PL_x, C7PL_y + 700), thickness=2, color=(0, 255, 0), tipLength=0.02)
#         cv2.line(image, (CSVL_x, CSVL_y), (CSVL_x, CSVL_y - 700), thickness=2, color=(0, 0, 255))

#     # print(avt_distance, avt_chest_distance, avt_waist_distance)

#     cv2.imwrite("./avt.jpg", image)


    
    avt_chest_distance = chest_x - C7PL_x
    avt_waist_distance = waist_x - CSVL_x
    # 全部转为python基本数据类型
    avt_chest_distance = float(avt_chest_distance)
    avt_waist_distance = float(avt_waist_distance)
    chest_x = int(chest_x)
    chest_y = int(chest_y)
    waist_x = int(waist_x)
    waist_y = int(waist_y)

    return [avt_chest_distance, avt_waist_distance], [[chest_x, chest_y], [waist_x, waist_y],[CSVL_x, CSVL_y],[C7PL_x, C7PL_y]]

# 冠状面平衡
def find_coronal_balance(image):


    background = image[:, :, 0]

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)  # 计算这个轮廓的周长
        area = cv2.contourArea(c)

        if area < 150:
            continue

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
        # # print(rotatedRect[2])
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        # # print("box:",box)
        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下

        line_point = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        lines.append(line_point)

    CSVL_x, CSVL_y, C7PL_x, C7PL_y = 0, 0, 0, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue

        CSVL_x, CSVL_y = find_CSVL(c)
        break

    for c in reversed(contours):
        area = cv2.contourArea(c)
        if area < 150:
            continue
        C7PL_x, C7PL_y = find_C7PL(c)
        break

    coronal_balance_distance = C7PL_x - CSVL_x

# # 注释
#     if C7PL_x == CSVL_x:
#         cv2.line(image, (CSVL_x, CSVL_y + 100), (C7PL_x, C7PL_y - 100), thickness=2, color=(0, 0, 255))
#     else:
#         cv2.arrowedLine(image, (C7PL_x, C7PL_y), (C7PL_x, C7PL_y + 450), thickness=2, color=(0, 255, 0), tipLength=0.02)
#         cv2.line(image, (CSVL_x, CSVL_y), (CSVL_x, CSVL_y - 400), thickness=2, color=(0, 0, 255))

#     cv2.putText(image, "Coronal Balance:" + str(round(coronal_balance_distance, 2)), (0, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     cv2.imwrite("./coronal_banlance.jpg", image)



    coronal_balance_distance = float(coronal_balance_distance)
    
    CSVL_x = int(CSVL_x)
    CSVL_y = int(CSVL_y)
    C7PL_x = int(C7PL_x)
    C7PL_y = int(C7PL_y)

    return coronal_balance_distance,[[CSVL_x, CSVL_y],[C7PL_x, C7PL_y]]

## 找两个顶锥
def find_apex_2(path, result_path):
    # path = './output/447.jpg'
    image = cv2.imread(path)

    background = image[:, :, 0]

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        perimeter = cv2.arcLength(c, True)  # 计算这个轮廓的周长
        area = cv2.contourArea(c)

        if area < 150:
            continue

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
        # # print(rotatedRect[2])
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)

        # # print("box:",box)
        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下

        line_point = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        lines.append(line_point)

    CSVL_x, CSVL_y, C7PL_x, C7PL_y = 0, 0, 0, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue

        CSVL_x, CSVL_y = find_CSVL(c)
        break

    for c in reversed(contours):
        area = cv2.contourArea(c)
        if area < 150:
            continue
        C7PL_x, C7PL_y = find_C7PL(c)
        break

    chest_x, chest_y, waist_x, waist_y = find_chest_waist_apex(CSVL_x, C7PL_x, threshold)

    waist_rectangle = find_apex_rectangle(waist_x, waist_y, width, lines)
    chest_rectangle = find_apex_rectangle(chest_x, chest_y, width, lines)

    chest_box = cv2.boxPoints(cv2.minAreaRect(contours[chest_rectangle]))
    chest_box = np.int_(chest_box)
    cv2.drawContours(image, [chest_box], 0, (0, 255, 0), 2)

    waist_box = cv2.boxPoints(cv2.minAreaRect(contours[waist_rectangle]))
    waist_box = np.int_(waist_box)
    cv2.drawContours(image, [waist_box], 0, (0, 0, 255), 2)

    # print(avt_distance, avt_chest_distance, avt_waist_distance)

    cv2.imwrite(result_path, image)


def calc_abc_from_line_2d(point1, point2):
    x0, y0 = point1
    x1, y1 = point2
    a = y0 - y1
    b = x1 - x0
    c = x0 * y1 - x1 * y0
    return a, b, c

def get_line_cross_point(point1, point2, point3, point4):
    a0, b0, c0 = calc_abc_from_line_2d(point1, point2)
    a1, b1, c1 = calc_abc_from_line_2d(point3, point4)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    cross_point = (int(x), int(y))
    return cross_point


# 返回值为两个集合,一个为点集,一个为box集
def draw_cobb(lines, box_list, begin_index, end_index, image):
    cr_point = get_line_cross_point(lines[begin_index][0], lines[begin_index][1], lines[end_index][2],
                                    lines[end_index][3])

    # cv2.line(image, lines[begin_index][0], cr_point, thickness=1, color=(0, 0, 255))
    # cv2.line(image, lines[begin_index][1], cr_point, thickness=1, color=(0, 0, 255))
    # cv2.line(image, lines[end_index][2], cr_point, thickness=1, color=(0, 0, 255))
    # cv2.line(image, lines[end_index][3], cr_point, thickness=1, color=(0, 0, 255))

    # cv2.drawContours(image, [box_list[begin_index]], 0, (0, 0, 255), 2)
    # cv2.drawContours(image, [box_list[end_index]], 0, (0, 0, 255), 2)
    #  先将所有的点的值转换为基本数据类型
    cr_point = [int(cr_point[0]), int(cr_point[1])]
    begin0 = [int(lines[begin_index][0][0]), int(lines[begin_index][0][1])]
    begin1 = [int(lines[begin_index][1][0]), int(lines[begin_index][1][1])]
    end2 = [int(lines[end_index][2][0]), int(lines[end_index][2][1])]
    end3 = [int(lines[end_index][3][0]), int(lines[end_index][3][1])]
    begin_box = [[int(box_list[begin_index][0][0]), int(box_list[begin_index][0][1])],
                 [int(box_list[begin_index][1][0]), int(box_list[begin_index][1][1])],
                 [int(box_list[begin_index][2][0]), int(box_list[begin_index][2][1])],
                 [int(box_list[begin_index][3][0]), int(box_list[begin_index][3][1])]]
    end_box = [[int(box_list[end_index][0][0]), int(box_list[end_index][0][1])],
                [int(box_list[end_index][1][0]), int(box_list[end_index][1][1])],
                [int(box_list[end_index][2][0]), int(box_list[end_index][2][1])],
                [int(box_list[end_index][3][0]), int(box_list[end_index][3][1])]]
    return [[cr_point, begin0, begin1, end2, end3], [begin_box, end_box]]
    
def find_rsh_point(output):
    # cv2.imwrite('output.jpg',output)
    output_left = output[:,0:output.shape[1]//3]
    output_right = output[:,output.shape[1]//3*2:]
    # 找出位于最左边的点
    non_zero_indices = np.where(output_left!=0)[1]
    if non_zero_indices.size == 0:
        # print('no rsh found, left')
        return [0,0],[0,0]
    left_min = np.min(non_zero_indices)
    left_point = [np.where(output_left!=0)[0][np.where(output_left!=0)[1]==left_min][0],left_min]
    # 找出位于最右边的点
    non_zero_indices = np.where(output_right!=0)[1]
    if non_zero_indices.size == 0:
        # print('no rsh found, right')
        return [0,0],[0,0]
    right_max = np.max(non_zero_indices)
    right_point = [np.where(output_right!=0)[0][np.where(output_right!=0)[1]==right_max][0],right_max+output.shape[1]//3*2]
    return left_point,right_point

# 找到cobb角
def find_Cobb_new(image):
    # path = 'try/week8/199.jpg'
    # result_path = 'try/week8/199_res.jpg'
    res_detail = []
    background = image[:, :, 0]

    _, threshold = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)
    hight, width = threshold.shape

    contours, cnt = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # up: 长方体上端的角度，bottom:长方体下端的角度
    up = []
    bottom = []
    rotate = []
    lines = []
    up_rotate = []
    bottom_rotate = []
    box_list = []

    contours_count = 0
    for c in contours:
        contours_count += 1
        perimeter = cv2.arcLength(c, True)  # 计算这个轮廓的周长
        area = cv2.contourArea(c)

        if (area < 150 and contours_count > 8) or (area < 350 and contours_count <= 4) or (
                area < 300 and 4 < contours_count <= 8):
            # print("find cobb area:", path, contours_count, perimeter, area)
            continue
        
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        rotatedRect = cv2.minAreaRect(c)  # 计算最小外接矩形
        # # print(rotatedRect[2])
        box = cv2.boxPoints(rotatedRect)
        box = np.int_(box)
        box_list.append(box)
        # # print("box:",box)
        if rotatedRect[2] > 45:
            x1, y1 = box[0]  # 左上
            x2, y2 = box[1]  # 右上
            x3, y3 = box[2]  # 右下
            x4, y4 = box[3]  # 左下
            rotate.append(1)
        else:
            x1, y1 = box[1]  # 左上
            x2, y2 = box[2]  # 右上
            x3, y3 = box[3]  # 右下
            x4, y4 = box[0]  # 左下
            rotate.append(0)
        up_k = (y2 - y1) / (x2 - x1)
        bottom_k = (y3 - y4) / (x3 - x4)
        up_angle = np.rad2deg(np.arctan2(y1 - y2, x1 - x2))
        bottom_angle = np.rad2deg(np.arctan2(y4 - y3, x4 - x3))
        # print(up_angle,bottom_angle)

        # up.append(math.fabs(up_angle))
        # bottom.append(math.fabs(bottom_angle))
        up.append(up_angle)
        bottom.append(bottom_angle)

        line_point = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        lines.append(line_point)

    # print("count area:", path, len(lines))

    lines.reverse()
    rotate.reverse()
    up.reverse()
    bottom.reverse()
    box_list.reverse()

    bone = ['C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4',
            'L5', 'S1']
    dic = {
        'C6': 0, 'C7': 1, 'T1': 2, 'T2': 3, 'T3': 4, 'T4': 5, 'T5': 6, 'T6': 7, 'T7': 8, 'T8': 9, 'T9': 10, 'T10': 11, 'T11': 12, 'T12': 13, 'L1': 14, 'L2': 15, 'L3': 16, 'L4': 17, 'L5': 18, 'S1': 19, 'S2': 20
    }
    if len(lines) == 20:
        bone = ['C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3',
                'L4',
                'L5', 'S1']
    elif len(lines) == 21:
        bone = ['C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3',
                'L4', 'L5', 'S1', 'S2']
    bone_box = {
    }
    for i in range(len(lines)):
        bone_box[bone[i]] = [[int(box_list[i][0][0]), int(box_list[i][0][1])], [int(box_list[i][1][0]), int(box_list[i][1][1])], [int(box_list[i][2][0]), int(box_list[i][2][1])], [int(box_list[i][3][0]), int(box_list[i][3][1])]]
    # for i in range(len(lines) - 1):
    #     print(lines[i], rotate[i], up[i], bottom[i], bone[i])

    i = 0
    while up[i] == 180 and i < len(lines) - 1:
        i += 1

    up_for_loop, bottom_for_loop = 180 - abs(up[i]), 180 - abs(bottom[i])
    begin_index = i
    last_180_index = i - 1
    find_down = False
    find_zero = False
    end_index = -1
    num_cob = 0
    up_bone, down_bone = [], []
    if up[i] > 0:
        pos_or_neg = "pos"
    else:
        pos_or_neg = "neg"

    # print("begin:", bone[begin_index], up[begin_index])
    i += 1
    while i < len(lines) - 1:
        if up[i] == 180:
            # 下端椎需要与上端椎间隔一个椎体以上
            if i - begin_index == 1:
                i += 1
                continue
            end_index = i
            bottom_for_loop = 0
            find_zero = True
            # break
        elif up[i] < 0:
            if pos_or_neg == "neg":
                if not find_zero:
                    if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                        up_for_loop = 180 - abs(up[i])
                        begin_index = i
                else:
                    break
            else:
                end_index = i
                bottom_for_loop = 180 - abs(bottom[i])
                break
        else:  # up[i]>0
            if pos_or_neg == "pos":
                if not find_zero:
                    if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                        up_for_loop = 180 - abs(up[i])
                        begin_index = i
                else:
                    break
            else:
                end_index = i
                bottom_for_loop = 180 - abs(bottom[i])
                break
        i += 1

    # print("end:", bone[end_index], up[end_index])
    # print("current:", bone[i])

    if end_index == -1:
        # print("only one direction draw:")
        if last_180_index == -1:
            for i in range(len(lines)):
                if bottom[i] == 180:
                    end_index = i

            res_detail.append(draw_cobb(lines, box_list, begin_index, end_index, image))
            up_bone.append(begin_index)
            down_bone.append(end_index)
        else:
            res_detail.append(draw_cobb(lines, box_list, 0, begin_index, image))
            up_bone.append(0)
            down_bone.append(begin_index)
        num_cob += 1

    # i += 1

    while i < len(lines) - 1:
        find_zero = False
        # print(bone[i], up[i], pos_or_neg)
        if up[i] == 180:
            j = i + 1
            if j < len(lines) - 1 and (
                    up[j] < 0 and pos_or_neg == 'pos' or up[j] > 0 and pos_or_neg == 'neg' or up[j] == 180):
                i += 1
                continue
            # i += 1
            # continue

            # print(bone[i], " up[i]=0,draw:", bone[begin_index], up[begin_index], bone[end_index], bottom_for_loop)
            num_cob += 1
            res_detail.append(draw_cobb(lines, box_list, begin_index, end_index, image))
            up_bone.append(begin_index)
            down_bone.append(end_index)

            if bottom_for_loop == 0:
                while up[i] == 180 and i < len(lines) - 1:
                    i += 1
                up_for_loop, bottom_for_loop = 180 - abs(up[i]), 180 - abs(bottom[i])
                begin_index = i
                if up[i] > 0:
                    pos_or_neg = "pos"
                else:
                    pos_or_neg = "neg"

                i += 1
                end_index = -1
                while i < len(lines) - 1:
                    if up[i] == 180:
                        if i - begin_index == 1:
                            i += 1
                            continue
                        end_index = i
                        bottom_for_loop = 0
                        break
                    elif up[i] < 0:
                        if pos_or_neg == "neg":
                            if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                                up_for_loop = 180 - abs(up[i])
                                begin_index = i
                        else:
                            end_index = i
                            bottom_for_loop = 180 - abs(bottom[i])
                            break
                    else:
                        if pos_or_neg == "pos":
                            if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                                up_for_loop = 180 - abs(up[i])
                                begin_index = i
                        else:
                            end_index = i
                            bottom_for_loop = 180 - abs(bottom[i])
                            break
                    i += 1
                    continue
            # find_down = False
            else:
                begin_index = end_index
                while bottom[i] == 180 and i < len(lines) - 1:
                    i += 1
                end_index = i
                bottom_for_loop = 180 - abs(bottom[i])
                if pos_or_neg == "pos":
                    pos_or_neg = "neg"
                else:
                    pos_or_neg = "pos"

        elif up[i] < 0:
            if pos_or_neg == "neg":
                num_cob += 1
                res_detail.append(draw_cobb(lines, box_list, begin_index, end_index, image))
                up_bone.append(begin_index)
                down_bone.append(end_index)

                # find_down = False
                if bottom_for_loop == 0:
                    while up[i] == 180 and i < len(lines) - 1:
                        i += 1
                    up_for_loop, bottom_for_loop = 180 - abs(up[i]), 180 - abs(bottom[i])
                    begin_index = i
                    if up[i] > 0:
                        pos_or_neg = "pos"
                    else:
                        pos_or_neg = "neg"

                    i += 1
                    end_index = -1
                    while i < len(lines) - 1:
                        if up[i] == 180:
                            # 下端椎需要与上端椎间隔一个椎体以上
                            if i - begin_index == 1:
                                i += 1
                                continue
                            end_index = i
                            bottom_for_loop = 0
                            find_zero = True
                            # break
                        elif up[i] < 0:
                            if pos_or_neg == "neg":
                                if not find_zero:
                                    if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                                        up_for_loop = 180 - abs(up[i])
                                        begin_index = i
                                else:
                                    break
                            else:
                                end_index = i
                                bottom_for_loop = 180 - abs(bottom[i])
                                break
                        else:
                            if pos_or_neg == "pos":
                                if not find_zero:
                                    if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                                        up_for_loop = 180 - abs(up[i])
                                        begin_index = i
                                else:
                                    break
                            else:
                                end_index = i
                                bottom_for_loop = 180 - abs(bottom[i])
                                break
                        i += 1
                        continue
                else:
                    begin_index = end_index
                    end_index = i
                    bottom_for_loop = 180 - abs(bottom[i])
                    pos_or_neg = "pos"
            else:
                if 180 - abs(bottom[i]) > bottom_for_loop and up[i+1]!=180:
                    bottom_for_loop = 180 - abs(bottom[i])
                    end_index = i
        else:
            if pos_or_neg == "pos":
                # print(bone[i], " up[i]>0,draw:", bone[begin_index], up[begin_index], bone[end_index], bottom_for_loop)
                num_cob += 1
                res_detail.append(draw_cobb(lines, box_list, begin_index, end_index, image))
                up_bone.append(begin_index)
                down_bone.append(end_index)

                # find_down = False
                if bottom_for_loop == 0:
                    while up[i] == 180 and i < len(lines) - 1:
                        i += 1
                    up_for_loop, bottom_for_loop = 180 - abs(up[i]), 180 - abs(bottom[i])
                    begin_index = i
                    if up[i] > 0:
                        pos_or_neg = "pos"
                    else:
                        pos_or_neg = "neg"

                    i += 1
                    end_index = -1
                    while i < len(lines) - 1:
                        if up[i] == 180:
                            # 下端椎需要与上端椎间隔一个椎体以上
                            if i - begin_index == 1:
                                i += 1
                                continue
                            end_index = i
                            bottom_for_loop = 0
                            find_zero = True
                            # break
                        elif up[i] < 0:
                            if pos_or_neg == "neg":
                                if not find_zero:
                                    if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                                        up_for_loop = 180 - abs(up[i])
                                        begin_index = i
                                else:
                                    break
                            else:
                                end_index = i
                                bottom_for_loop = 180 - abs(bottom[i])
                                break
                        else:  # up[i]>0
                            if pos_or_neg == "pos":
                                if not find_zero:
                                    if 180 - abs(up[i]) > up_for_loop and up[i+1]!=180:
                                        up_for_loop = 180 - abs(up[i])
                                        begin_index = i
                                else:
                                    break
                            else:
                                end_index = i
                                bottom_for_loop = 180 - abs(bottom[i])
                                break
                        i += 1
                    continue
                else:
                    begin_index = end_index
                    end_index = i
                    bottom_for_loop = 180 - abs(bottom[i])
                    pos_or_neg = "neg"
            else:
                if 180 - abs(bottom[i]) > bottom_for_loop and up[i+1]!=180:
                    bottom_for_loop = 180 - abs(bottom[i])
                    end_index = i
        i += 1


    # print(end_index, len(lines))
    if end_index != -1 and end_index != len(lines) - 1:
        # print("end draw:", bone[begin_index], up[begin_index], bone[end_index], bottom_for_loop)
        res_detail.append(draw_cobb(lines, box_list, begin_index, end_index, image))
        up_bone.append(begin_index)
        down_bone.append(end_index)
        num_cob += 1

    result = []
    has_chest = False
    switch = {
        'C6': '颈胸弯', 'C7': '颈胸弯', 'T1': '颈胸弯', 'T2': '上胸弯', 'T3': '上胸弯', 'T4': '上胸弯', 'T5': '上胸弯', 'T6': '胸弯', 'T7': '胸弯', 'T8': '胸弯', 'T9': '胸弯', 'T10': '胸弯', 'T11': '胸弯', 'T12': '胸腰弯', 'L1': '胸腰弯', 'L2': '腰弯', 'L3': '腰弯', 'L4': '腰弯', 'L5': '腰弯', 'S1': '腰弯', 'S2': '腰弯'
    }
    for i in range(len(up_bone)):
        # 先判断是左弯还是右弯
        up_point1 = res_detail[i][0][1]
        up_point2 = res_detail[i][0][2]
        # 避免分母为0
        if up_point2[0] - up_point1[0] == 0:
            up_k = 999999
        else:
            up_k = (up_point2[1] - up_point1[1]) / (up_point2[0] - up_point1[0])
        if up_k > 0:
            direction = "右弯"
        else:
            direction = "左弯"
        # 这里还要加一个顶椎,以及顶椎的旋转程度,还得判断是左弯还是右弯
        left_min_index = up_bone[i]
        right_max_index = down_bone[i]
        for j in range(up_bone[i],down_bone[i]+1):
            if min(box_list[j][0][0],box_list[j][3][0]) < min(box_list[left_min_index][0][0],box_list[left_min_index][3][0]):
                left_min_index = j
            if max(box_list[j][1][0],box_list[j][2][0]) > max(box_list[right_max_index][1][0],box_list[right_max_index][2][0]):
                right_max_index = j
        if direction == "左弯":
            apex_index = right_max_index
        else :
            apex_index = left_min_index
        apex_type = switch[bone[apex_index]]
        if apex_type == '胸弯':
            if has_chest == False:
                has_chest = True
            else:
                apex_type = '胸弯2'
        # 找到顶椎的bounding box
        apex_box = box_list[apex_index]
        apex_min_x = min(apex_box[0][0],apex_box[3][0])
        apex_max_x = max(apex_box[1][0],apex_box[2][0])
        apex_min_y = min(apex_box[0][1],apex_box[1][1])
        apex_max_y = max(apex_box[2][1],apex_box[3][1])
        apex_img = image[apex_min_y:apex_max_y,apex_min_x:apex_max_x]
        print("apex_img:",apex_img.shape)

        # 这边需要集成一下Nash_Moe的计算, 使用apex_img
        Nash_Moe = VR_eval(apex_img)
        Nash_Moe = int(Nash_Moe)
        # Nash_Moe = 1
        # result的形式为：[类型, 方向，上端椎，下端椎，顶椎，Nash_Moe, Cobb角度]
        # 类型分为: 颈胸弯C7～T1, 上胸弯T2-T5 , 胸弯T6-T11 , 胸弯2 T6-T11(第二块顶椎) , 胸腰弯T12～L1, 腰弯L2～S2
        result.append([apex_type, direction, bone[up_bone[i]], bone[down_bone[i]], bone[apex_index], Nash_Moe, float(round(360 - abs(up[up_bone[i]]) - abs(bottom[down_bone[i]]), 2))])
    # cv2.imwrite("./result.jpg", image)
    return result,res_detail,bone_box


# 辅助函数
def max_num(a,b):
    if a > b:
        return a
    else:
        return b
# 辅助函数
def min_num(a,b):
    if a < b:
        return a
    else:
        return b
    
# 这是find_RSH的辅助函数,用于找出肩锁关节和表皮组织的交点
def process_point_set(point_set,x):
    # 按照x坐标大小排序
    point_set.sort(key = lambda x:x[0])
    # 对于同一x坐标的点,取y坐标最大的点
    point_set_processed = []
    point_set_processed.append(point_set[0])
    for i in range(1,len(point_set)-1):
        if point_set[i][0] == point_set[i+1][0]:
            if point_set[i][1] < point_set[i+1][1]:
                point_set_processed[len(point_set_processed)-1] = point_set[i+1]
        else:  
            point_set_processed.append(point_set[i])
    # 遍历point_set_processed,做线性插值,让x坐标连续
    point_set_processed_new = []
    result_y = point_set_processed[0][1]
    for i in range(len(point_set_processed)-1):
        result_y = point_set_processed[i][1]
        if(point_set_processed[i][0] == x):
            return point_set_processed[i]
        if point_set_processed[i][0] == point_set_processed[i+1][0]-1:
            point_set_processed_new.append(point_set_processed[i])
        else:
            for j in range(point_set_processed[i][0]+1,point_set_processed[i+1][0]):
                y = (point_set_processed[i][1] + (point_set_processed[i+1][1]-point_set_processed[i][1])*(j-point_set_processed[i][0])//(point_set_processed[i+1][0]-point_set_processed[i][0]))
                point_set_processed_new.append([j,y])
                if j == x:
                    return [j,y]
    return [x,result_y]

# 辅助函数找到肩锁关节与表皮组织的交点,出错我就返回[0,0]
def find_tissue_point(raw_image,left_point,right_point):
    # 灰度化图像
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, threshold = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    # 轮廓检测
    contours,_ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大轮廓
    max_contour = max(contours, key = cv2.contourArea)
    # 找到左点集
    left_point_set = []
    for point in max_contour:
        if point[0][0] >= max_num(0,left_point[1] -50) and point[0][0] <= min_num(image.shape[1],left_point[1]+50) and point[0][1] >= max_num(0,left_point[0]-400) and point[0][1] <= min_num(image.shape[0],left_point[0]+50):
            left_point_set.append(point[0])
    # 找到右点集
    right_point_set = []
    for point in max_contour:
        if point[0][0] >= max_num(0,right_point[1] -50) and point[0][0] <= min_num(image.shape[1],right_point[1]+50) and point[0][1] >= max_num(0,right_point[0]-400) and point[0][1] <= min_num(image.shape[0],right_point[0]+50):
            right_point_set.append(point[0])
    if len(left_point_set) == 0 or len(right_point_set) == 0:
        print('Failed to find point set, maybe we find the wrong point')
        return [0,0],[0,0]
    left_intersection = process_point_set(left_point_set,left_point[1])
    right_intersection = process_point_set(right_point_set,right_point[1])
    return left_intersection,right_intersection

def process_output(output):
    # 先找output的轮廓
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 保留面积最大的两个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    # 找出位于轮廓范围内的所有像素点
    mask = np.zeros_like(output)
    cv2.drawContours(mask, contours, -1, (255), -1)
    output = cv2.bitwise_and(output, mask)
    return output

# 函数: find_rsh
# 函数说明：找到图片的肩高
# 参数说明：
# 返回值：肩高，以及左右两个肩锁关节和表皮组织的交点,格式为[[左肩锁关节],[右肩锁关节],[左表皮组织],[右表皮组织]]
def find_rsh(image):
    raw_image = image
    pixels = find_centermeter_per_pixel(raw_image)
    image,shift = split_image(raw_image)
    output = get_Unet_output(image)
    mask = process_output(output)
    left_point,right_point = find_rsh_point(mask)
    # 将image 进行canney边缘检测
    edges = cv2.Canny(image,30,120)
    # edges经过mask处理
    edges = cv2.bitwise_and(edges,mask)
    left_point_canny,right_point_canny = find_rsh_point(edges)
    # 计算两次点的距离
    left_point_distance = abs(left_point[0]-left_point_canny[0]) + abs(left_point[1]-left_point_canny[1])
    right_point_distance = abs(right_point[0]-right_point_canny[0]) + abs(right_point[1]-right_point_canny[1])
    # 如果距离小于10, 则left_point,right_point取canny的点
    if left_point_distance < 10:
        left_point = left_point_canny
    if right_point_distance < 10:
        right_point = right_point_canny

    # 画出左右两个点
    # output_img = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
    # output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)
    # cv2.circle(output_img,(left_point[1],left_point[0]),25,(255,0,0),-1)
    # cv2.circle(output_img,(right_point[1],right_point[0]),25,(255,0,0),-1)
    # output_img = Image.fromarray(output_img)
    # output_img.save(os.path.join(result_folder,os.path.basename(input_path)))
    # 将左右两个点加上偏移
    left_point[1] += shift[0]
    right_point[1] += shift[0]
    left_point[0] += shift[2]
    right_point[0] += shift[2]
    # 找到左右两个tissue点
    left_tissue_point,right_tissue_point = find_tissue_point(raw_image,left_point,right_point)
    # 画出从肩锁关节到表皮组织的直线
    rsh = (right_tissue_point[1]-left_tissue_point[1]) / pixels * 10

    # cv2.circle(raw_image,(left_point[1],left_point[0]),15,(0,255,0),-1)
    # cv2.circle(raw_image,(right_point[1],right_point[0]),15,(0,255,0),-1)
    # cv2.circle(raw_image,(left_tissue_point[0],left_tissue_point[1]),15,(0,255,0),-1)
    # cv2.circle(raw_image,(right_tissue_point[0],right_tissue_point[1]),15,(0,255,0),-1)
    # cv2.line(raw_image, (left_point[1], left_point[0]), (left_tissue_point[0], left_tissue_point[1]), (0, 0, 255), 10)
    # cv2.line(raw_image, (right_point[1], right_point[0]), (right_tissue_point[0], right_tissue_point[1]), (0, 0, 255), 10)
    # # # 画出两条水平线
    # cv2.line(raw_image, (left_tissue_point[0], left_tissue_point[1]), (right_tissue_point[0], left_tissue_point[1]), (0, 0, 255), 10)
    # cv2.line(raw_image, (left_tissue_point[0], right_tissue_point[1]), (right_tissue_point[0], right_tissue_point[1]), (0, 0, 255), 10)
    # cv2.putText(raw_image, "RSH: {:.3} cm".format(float(rsh)), (min(left_tissue_point[0],right_tissue_point[0])-60,min(left_tissue_point[1],right_tissue_point[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # pil_img = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    # pil_img.save(os.path.join(rsh_folder,os.path.basename(input_path)))
    # print('Done! RSH:',rsh)

    # 转为基本数据类型
    rsh = float(rsh)
    left_point = [int(left_point[1]),int(left_point[0])]
    right_point = [int(right_point[1]),int(right_point[0])]
    left_tissue_point = [int(left_tissue_point[0]),int(left_tissue_point[1])]
    right_tissue_point = [int(right_tissue_point[0]),int(right_tissue_point[1])]
    return rsh,[left_point,right_point,left_tissue_point,right_tissue_point]


# 辅助函数
def find_clavicle_point(output):
    output_left = output[:,0:output.shape[1]//3]
    output_right = output[:,output.shape[1]//3*2:]
    # 找出左边像素值不为0的高度最小的点
    non_zero_indices = np.where(output_left!=0)[0]
    if non_zero_indices.size == 0:
        # print('no clavicle found, left')
        return [0,0],[0,0]
    left_min = np.min(non_zero_indices)
    left_point = [left_min,np.where(output_left!=0)[1][np.where(output_left!=0)[0]==left_min][0]]
    # 找出右边像素值不为0的高度最小的点
    non_zero_indices = np.where(output_right!=0)[0]
    if non_zero_indices.size == 0:
        # print('no clavicle found, right')
        return [0,0],[0,0]
    right_min = np.min(non_zero_indices)
    right_point = [right_min,np.where(output_right!=0)[1][np.where(output_right!=0)[0]==right_min][0]+output.shape[1]//3*2]
    return left_point,right_point

# 函数: find_clavicle_angle
# 函数说明：找到图片的锁骨角度
# 参数说明：
# 返回值：angle, 以及左右两个锁骨关节的坐标,格式为[[左锁骨关节],[右锁骨关节]]
def find_clavicle_angle(image):
    raw_image = image
    image,shift = split_image(raw_image)
    output = get_Unet_output(image)
    mask = process_output(output)
    left_point,right_point = find_clavicle_point(mask)
    # 边缘检测
    edges = cv2.Canny(image,30,100)
    # edges经过mask处理
    edges = cv2.bitwise_and(edges,mask)
    left_point_canny,right_point_canny = find_clavicle_point(edges)
    # 如果left_point_canny的值不为[0,0] 则left_point取left_point_canny
    if left_point_canny != [0,0]:
        left_point = left_point_canny
    # 如果right_point_canny的值不为[0,0] 则right_point取right_point_canny
    if right_point_canny != [0,0]:
        right_point = right_point_canny

    # output_img = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
    # output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2RGB)
    # cv2.circle(output_img,(left_point[1],left_point[0]),2,(255,0,0),-1)
    # cv2.circle(output_img,(right_point[1],right_point[0]),2,(255,0,0),-1)
    # output_img = Image.fromarray(output_img)
    # output_img.save(os.path.join(result_folder,os.path.basename(input_path)))

    # 将左右两个点加上偏移
    left_point[1] += shift[0]
    right_point[1] += shift[0]
    left_point[0] += shift[2]
    right_point[0] += shift[2]
    # 计算左右两个点的角度
    angle = np.arctan((right_point[0]-left_point[0])/(right_point[1]-left_point[1]))/np.pi*180
    # 判断左右点的y坐标,选择较小的y坐标
    if left_point[0] > right_point[0]:
        x = right_point[0]
        y = right_point[1]
    else:
        x = left_point[0]
        y = left_point[1]
    # 计算左右点的斜率
    k = (right_point[0] - left_point[0]) / (right_point[1] - left_point[1])
    # 计算角度
    angle = np.arctan(k) * 180 / np.pi
    # 在图像中标出角度
    # 保存图像

    # cv2.line(raw_image, (left_point[1], x), (right_point[1], x), (0, 0, 255), 2)
    # # 画出连接left_point和right_point的直线
    # cv2.line(raw_image, (left_point[1], left_point[0]), (right_point[1], right_point[0]), (0, 0, 255), 2)
    # cv2.putText(raw_image, "Angle: {:.3} degree".format(float(angle)), (max(y-60,0),x-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # pil_img = Image.fromarray(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    # pil_img.save(os.path.join(clavice_folder,os.path.basename(input_path)))

    print('Done! angle:',angle)
    angle = float(angle)
    left_point = [int(left_point[1]),int(left_point[0])]
    right_point = [int(right_point[1]),int(right_point[0])]
    return angle, [left_point,right_point]


# 由于cv2无法读取中文路径, 所以需要将中文路径转为英文路径
def open_img(path):
    img = Image.open(path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


# 由于cv2无法保存中文路径, 所以需要将中文路径转为英文路径
def save_img(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.save(path)


def get_contours(recovered_img):
    # 寻找recover_img的轮廓
    gray = cv2.cvtColor(recovered_img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 150)
    dilated_edged = cv2.dilate(edged.copy(), None, iterations=2)
    # 保存图片
    # cv2.imwrite('./VR/temp/edged.jpg', edged)
    contours = cv2.findContours(dilated_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 3000]
    # # 在img中画出轮廓
    # image = np.zeros_like(img)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    # i = 0
    # for contour in contours:
    #     # 找到轮廓的矩形
    #     i += 1
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.putText(img, str(i), (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     # 画出矩形
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # # 保存图片
    # cv2.imwrite('./VR/temp/raw_contours.jpg', img)
    # cv2.imwrite('./VR/temp/contours.jpg', image)
    # # 灰度化
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)

    #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # 保存图片
    #cv2.imwrite('./contours.jpg', img)
    return contours


def find_apex(contours,csvl):
    index = 7
    distance = 0
    for i in range(8):
        contour = contours[i + 7]
        d = 0
        for point in contour:
            d = max(d, abs(point[0][0] - csvl[0]))
        if d > distance:
            distance = d
            index = i + 7
    print("apex index: ",index)
    return index

def find_center(contour):
    # 计算轮廓的几何矩
    M = cv2.moments(contour)

    # 使用几何矩计算轮廓的中心点
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return [cX, cY]

def find_cut_length(contours):
    # 从第4个轮廓开始
    button = contours[2]
    # 得到第12个轮廓
    top = contours[12]
    # 计算button和top的中心点
    button_center = find_center(button)
    top_center = find_center(top)
    # 计算button_center和top_center的y距离
    return (button_center[1] - top_center[1]) // 2 * 2

def get_cut_image(img,cut_length,center):
    # 计算以center为中心,长度为cut_length的正方形
    cut_length = cut_length + 256
    x1 = center[0] - cut_length // 2
    x2 = center[0] + cut_length // 2
    y1 = center[1] - cut_length // 2
    y2 = center[1] + cut_length // 2
    # 从img中截取这个正方形
    print("cut length: ",cut_length)
    if x1 < 0:
        x1 = 0
    if x2 > img.shape[1] - 64:
        x2 = img.shape[1] - 64
    if y1 < 64:
        y1 = 64
    if y2 > img.shape[0] - 64:
        y2 = img.shape[0] -64
    square = img[y1:y2, x1:x2]
    resized_image = cv2.resize(square, (512, 512))
    return square,resized_image,[x1,y1]


def find_box(contour):
    # 计算轮廓的最小外接矩形
    rect = cv2.minAreaRect(contour)
    # 计算最小外接矩形的四个顶点
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box

# 输入为识别恢复后的图片, 返回值 tts,[[left_point, right_point, chest_center, center],box,pixels]
def find_TTS(img,result):
    recovered_img = result
    # 得到每个椎体的轮廓
    contours = get_contours(recovered_img)
    pixels = find_centermeter_per_pixel(img)
    # 寻找csvl参考点
    if len(contours) < 1:
        print("No contours found")
        return 0,[[[0,0],[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0],[0,0]]]
    _,csvl_points = draw_CSVL(result)
    csvl = csvl_points[2]
    # 寻找胸弯顶椎index
    apex_index = find_apex(contours,csvl)
    # 寻找胸弯顶椎中心点
    center = find_center(contours[apex_index])

    # 寻找胸弯顶椎和胸弯底椎的长度
    cut_length = find_cut_length(contours)
    origin_cut, resize_cut,shift = get_cut_image(img,cut_length,[int((center[0] + csvl[0]) // 2),center[1]])
    # 保存resize_cut
    # resize_cut.save('./resize_cut.jpg')
    
    y = TTS_eval(resize_cut)

    left_point = [int(y[0]),center[1]]
    right_point = [int(y[1]),center[1]]
    # 将left_point和right_point转换为原始图片坐标
    left_point[0] = left_point[0] * origin_cut.shape[1] // 512 + shift[0]
    right_point[0] = right_point[0] * origin_cut.shape[1] // 512 + shift[0]
    chest_center = (left_point[0]+right_point[0])//2
    # 计算TTS
    tts = (chest_center - csvl[0])/pixels
    box = find_box(contours[apex_index])

    # # 可视化
    # image = img.copy()
    # # 画一条线,从(csvl[0],0) 到 csvl
    # cv2.line(image, (int(csvl[0]), 0), (int(csvl[0]), int(csvl[1])), (0, 255, 0), 2)
    # # 画一条线,从(chest_center,left_point[1] - 200) 到 (chest_center,left_point[1] + 200)
    # cv2.line(image, (chest_center, left_point[1] - 200), (chest_center, left_point[1] + 200), (255, 0, 0), 2)
    # # 画一条线,从(left_point[0],left_point[1]) 到 (right_point[0],right_point[1])
    # cv2.line(image, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (0, 255, 0), 2)
    # box = find_box(contours[apex_index])
    # distance = 0
    # for point in box:
    #     if abs(point[0] - csvl[0]) > distance:
    #         distance = abs(point[0] - csvl[0])
    #         apex = point
    # # 画一条线,从(apex[0],left_point[1] - 200) 到 (apex[0],left_point[1] + 200)
    # cv2.line(image, (apex[0], left_point[1] - 200), (apex[0], left_point[1] + 200), (0, 255, 0), 2)
    # # 将center画出来
    # cv2.circle(image, (center[0], center[1]), 5, (0, 0, 255), -1)
    # # 将chest_center画出来
    # cv2.circle(image, (chest_center, center[1]), 5, (0, 0, 255), -1)
    # # 将left_point画出来
    # cv2.circle(image, (left_point[0], left_point[1]), 5, (0, 0, 255), -1)
    # # 将right_point画出来
    # cv2.circle(image, (right_point[0], right_point[1]), 5, (0, 0, 255), -1)
    # # 保存图片
    # save_img(image, TTS_path)

    tts = float(tts)
    left_point = [int(left_point[0]),int(left_point[1])]
    right_point = [int(right_point[0]),int(right_point[1])]
    chest_center = [int(chest_center),int(center[1])]
    center = [int(center[0]),int(center[1])]
    box = [[int(point[0]),int(point[1])] for point in box]
    return tts,[[left_point, right_point, chest_center, center],box,pixels]



if __name__ == '__main__':
    print("start model_runtime/result/丁子航_psFEXF1.jpg, result: model_runtime/mid_result/丁子航_psFEXF1.jpg")
    # find_Cobb_new("./model_runtime/result/程诗画_G8a0vpX.jpg", "./model_runtime/mid_result/程诗画_G8a0vpX.jpg")
    find_Cobb_new(cv2.imread("./recovered.jpg"))