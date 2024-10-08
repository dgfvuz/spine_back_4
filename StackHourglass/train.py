import cv2
import numpy as np
from .model import StackHourglass
from .loss import StackHourglassLoss
import torch
import torch.optim as optim
from .model import StackHourglassDataset
from .config import MODEL_PATH, SMALL_MODEL_PATH
from PIL import Image
import torchvision
from torch.optim.lr_scheduler import StepLR
from sklearn.cluster import DBSCAN
from collections import Counter

# 开始训练, 输入参数为数据集所在文件夹
def start_training(dataset_folder, model_path = None):
    model = StackHourglass(2, 1, 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    criterion = StackHourglassLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=400, gamma=0.1)  # 初始化调度器
    model.train()
    dataset = StackHourglassDataset(dataset_folder)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print('Start Training')
    for epoch in range(800):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, heatmaps = data
            inputs, heatmaps = inputs.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.20f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        if epoch % 100 == 99:
            torch.save(model.state_dict(), 'model_' + str(epoch) + '.pth')
        scheduler.step()
    print('Finished Training')
    torch.save(model.state_dict(), 'model.pth')


# 输入cv2的image,现已弃用
def eval_small(image, points):
    model = StackHourglass(2, 9, 18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(SMALL_MODEL_PATH, map_location=device))
    # 判断image是否是灰度图
    if not len(image.shape) == 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1), "image is not gray"
    
    # 灰度化
    # 直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)
    # 图片宽和高
    h, w = image.shape

    img_list = []
    crop_list = []
    for i in range(0,9):
        center = [int((points[i*2][0] + points[i*2+1][0])/2) , int((points[i*2][1] + points[i*2+1][1])/2)]
        # 512*512
        x1 = center[0]-256
        x2 = center[0]+256
        y1 = center[1]-256
        y2 = center[1]+256
        # 生成随机数-8-8
        random_num1 = np.random.randint(-8,8)
        random_num2 = np.random.randint(-8,8)
        # 判断[y1+random_num1:y2+random_num1,x1+random_num2:x2+random_num2]是否超出边界,如果超出边界就平移
        if y1+random_num1 < 0:
            random_num1 = 0 - y1
        if y2+random_num1 > h:
            random_num1 = h - y2
        if x1+random_num2 < 0:
            random_num2 = 0 - x1
        if x2+random_num2 > w:
            random_num2 = w - x2
        img = image[y1+random_num1:y2+random_num1,x1+random_num2:x2+random_num2]
        crop_list.append([y1+random_num1,y2+random_num1,x1+random_num2,x2+random_num2])
        img_list.append(torchvision.transforms.ToTensor()(img).float().squeeze())
    # 将image转换为tensor
    image = torch.stack(img_list, dim=0).unsqueeze(0).to(device)

    model.to(device=device)
    model.eval()
    with torch.no_grad():
        output = model(image)

    
    points = []
    final_out = output[1].cpu()
    # 去除批次维度
    image_tensor = final_out.squeeze(0) 
    for i in range(18):
        channel_0 = image_tensor[i, :, :]
        heatmap_np = channel_0.numpy()
        max_idx_np = np.argmax(heatmap_np)
        max_coords_np = np.unravel_index(max_idx_np, heatmap_np.shape)
        y, x = max_coords_np
        x = (x*4 + crop_list[i//2][2])
        y = y*4 + crop_list[i//2][0]
        points.append([x,y])

    return points



# 输入cv2的image
def eval(image, raw_points = True):
    model = StackHourglass(2, 1, 18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
    # 判断image是否是灰度图
    if not len(image.shape) == 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1), "image is not gray"
    
    # 直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)

    # 图片宽和高
    h, w = image.shape
    h_factor = h / 1024
    # resize
    image = cv2.resize(cl1, (int(w / h_factor), 1024))

    # 图像两边补零让宽度能够被64
    final_w = int(w / h_factor) + (64 - int(w / h_factor) % 64)
    left = int((final_w - int(w / h_factor)) / 2)
    right = final_w - left - int(w / h_factor)
    image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
    image = Image.fromarray(image)
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    image = image.float().to(device)

    model.to(device=device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    # return output

    
    points = []
    final_out = output[1].cpu()
    # 去除批次维度
    image_tensor = final_out.squeeze(0) 
    for i in range(18):
        channel_0 = image_tensor[i, :, :]
        heatmap_np = channel_0.numpy()
        max_idx_np = np.argmax(heatmap_np)
        max_coords_np = np.unravel_index(max_idx_np, heatmap_np.shape)
        y, x = max_coords_np
        if raw_points:
            x = (x*4 - left) * h_factor
            y = y*4 * h_factor
        else:
            x = (x*4 - left)
            y = y*4
        points.append([x,y])

    return points




# @author: yasiare
def find_cluster(points,threshold = 5):
    """
        函数功能:
            寻找到points中的聚类, 并返回中心点

        输入:
            points: 一个包含了若干点的列表, 每个点是一个[x, y]坐标
            threshold: 阈值, 如果点超出了这个阈值则舍弃掉

        输出:
            point = [x, y] 输出聚类的中心
    """
    points = np.array(points)
    db = DBSCAN(eps=threshold, min_samples=2).fit(points)
    # 获取聚类标签
    labels = db.labels_

    label_counts = Counter(labels[labels != -1])

    if label_counts:
        largest_cluster_label = label_counts.most_common(1)[0][0]
        largest_cluster_points = points[labels == largest_cluster_label]
        center = np.mean(largest_cluster_points, axis=0)
        return center




# @author yasiare
# 改进评测方法: 使用调整亮度求相近值来确定点的位置
def imporved_eval(image, times = 5, raw_points = False, threshold=5):
    """
    函数功能:
    通过多次调整亮度以后送入模型
    
    参数:
    image: cv2读取的image
    times: 重复的次数
    raw_points: 是否转化为原来的点, False是高度为1024的点, true则转为原来的点

    返回值:
    list: 包含若干点(16个)的列表
    """

    # 加载模型
    model = StackHourglass(2, 1, 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
    model.to(device=device)
    model.eval()

    # 查看图像的亮度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)

    # 根据times计算需要的数组
    start_brightness = 45
    end_brightness = 66
    bright_list = [start_brightness + i * ((end_brightness - start_brightness - 1)//times) for i in range(times)]

    targets = []

    points_list_list = [[] for i in range(16)]
    for bright in bright_list:
        # 更改图像的亮度:
        change_bright = int(bright - brightness)
        changed_image = cv2.convertScaleAbs(image, alpha=1, beta= change_bright)
        # 判断image是否是灰度图
        if not len(changed_image.shape) == 2 and changed_image.shape[2] == 3:
            changed_image = cv2.cvtColor(changed_image, cv2.COLOR_BGR2GRAY)
        assert len(changed_image.shape) == 2 or (len(changed_image.shape) == 3 and changed_image.shape[2] == 1), "changed_image is not gray"
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(changed_image)

        # 图片宽和高
        h, w = changed_image.shape
        h_factor = h / 1024
        # resize
        changed_image = cv2.resize(cl1, (int(w / h_factor), 1024))

        # 图像两边补零让宽度能够被64
        final_w = int(w / h_factor) + (64 - int(w / h_factor) % 64)
        left = int((final_w - int(w / h_factor)) / 2)
        right = final_w - left - int(w / h_factor)
        changed_image = cv2.copyMakeBorder(changed_image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
        changed_image = Image.fromarray(changed_image)
        changed_image = torchvision.transforms.ToTensor()(changed_image).unsqueeze(0)
        changed_image = changed_image.float().to(device)

        with torch.no_grad():
            output = model(changed_image)
        # return output

        
        final_out = output[1].cpu()
        # 去除批次维度
        image_tensor = final_out.squeeze(0) 
        for i in range(16):
            channel_0 = image_tensor[i, :, :]
            heatmap_np = channel_0.numpy()
            max_idx_np = np.argmax(heatmap_np)
            max_coords_np = np.unravel_index(max_idx_np, heatmap_np.shape)
            y, x = max_coords_np
            x = (x*4 - left)
            y = y*4
            points_list_list[i].append([x,y])
    for points in points_list_list:
        cluster_point = find_cluster(points, threshold=threshold)
        if raw_points:
            cluster_point[0] = cluster_point[0] * h_factor
            cluster_point[1] = cluster_point[1] * h_factor
        targets.append(cluster_point)
    return targets


















# 输入cv2的image
def eval_big(image):
    model = StackHourglass(2, 1, 18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # 判断image是否是灰度图
    if not len(image.shape) == 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1), "image is not gray"
    
    # 直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)

    # 图片宽和高
    h, w = image.shape
    h_factor = h / 2048
    # resize
    image = cv2.resize(cl1, (int(w / h_factor), 2048))

    # 图像两边补零让宽度能够被64
    final_w = int(w / h_factor) + (64 - int(w / h_factor) % 64)
    left = int((final_w - int(w / h_factor)) / 2)
    right = final_w - left - int(w / h_factor)
    image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
    image = Image.fromarray(image)
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    image = image.float().to(device)

    model.to(device=device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    # return output

    
    points = []
    final_out = output[1].cpu()
    # 去除批次维度
    image_tensor = final_out.squeeze(0) 
    for i in range(18):
        channel_0 = image_tensor[i, :, :]
        heatmap_np = channel_0.numpy()
        max_idx_np = np.argmax(heatmap_np)
        max_coords_np = np.unravel_index(max_idx_np, heatmap_np.shape)
        y, x = max_coords_np
        x = (x*4 - left) * h_factor
        y = y*4 * h_factor
        # x = (x*4 - left)
        # y = y*4
        points.append([x,y])

    return points
