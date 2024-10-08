import base64
from io import BytesIO
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image
import torchvision
import time
# 残差连接模块
class ResidualModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualModule, self).__init__()
        
        # 第一层卷积 1x1
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        # 第二层卷积 3x3
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第三层卷积 1x1
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # 如果输入通道和输出通道不一致，需要调整输入的维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 将输入通过shortcut连接与卷积的结果相加
        out += self.shortcut(x)
        out = torch.relu(out)
        
        return out


# 下采样模块, 只进行一次下采样 变成1/2
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

# 上采样模块, 只进行一次上采样 变成2倍
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.upsample(out)
        return out




class Hourglass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Hourglass, self).__init__()
        self.jumpConnect1 = ResidualModule(in_channels, out_channels)
        self.jumpConnect2 = ResidualModule(256, out_channels)
        self.jumpConnect3 = ResidualModule(256, out_channels)
        self.jumpConnect4 = ResidualModule(256, out_channels)

        # 下采样
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # 上采样 采用反卷积
        self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

        # 卷积
        self.conv1 = nn.Sequential(
            ResidualModule(in_channels, 256),
            ResidualModule(256, 256),
            ResidualModule(256, 256)
        )
        self.conv2 = nn.Sequential(
            ResidualModule(256, 256),
            ResidualModule(256, 256),
            ResidualModule(256, 256)
        )
        self.conv3 = nn.Sequential(
            ResidualModule(256, 256),
            ResidualModule(256, 256),
            ResidualModule(256, 256)
        )
        self.conv4 = nn.Sequential(
            ResidualModule(256, 256),
            ResidualModule(256, 256),
            ResidualModule(256, 256)
        )

        self.middleConv = ResidualModule(256, out_channels)

        self.backCov1 = ResidualModule(out_channels, out_channels)
        self.backCov2 = ResidualModule(out_channels, out_channels)
        self.backCov3 = ResidualModule(out_channels, out_channels)
        self.backCov4 = ResidualModule(out_channels, out_channels)

    def forward(self, x):
        x_jump1 = self.jumpConnect1(x)
        x = self.downsample(x)
        x = self.conv1(x)

        x_jump2 = self.jumpConnect2(x)
        x = self.downsample(x)
        x = self.conv2(x)

        x_jump3 = self.jumpConnect3(x)
        x = self.downsample(x)
        x = self.conv3(x)

        x_jump4 = self.jumpConnect4(x)
        x = self.downsample(x)
        x = self.conv4(x)
        x = self.middleConv(x)
        x = self.backCov1(x)
        x = self.upsample1(x)
        x += x_jump4

        x = self.backCov2(x)
        x = self.upsample2(x)
        x += x_jump3

        x = self.backCov3(x)
        x = self.upsample3(x)
        x += x_jump2

        x = self.backCov4(x)
        x = self.upsample4(x)
        x += x_jump1

        return x
    
class StackHourglass(nn.Module):
    def __init__(self, num_stacks, in_channels, out_channels):
        super(StackHourglass, self).__init__()
        self.num_stacks = num_stacks
        self.hourglass1 = Hourglass(in_channels=256, out_channels=512)
        self.hourglass2 = Hourglass(in_channels=384, out_channels=512)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.residualCov1 = ResidualModule(in_channels=64, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residualCov2 = ResidualModule(in_channels=128, out_channels=128)
        self.residualCov3 = ResidualModule(in_channels=128, out_channels=128)
        self.residualCov4 = ResidualModule(in_channels=128, out_channels=256)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.outConv1 = nn.Conv2d(in_channels=256, out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        self.outConv2 = nn.Conv2d(in_channels=out_channels, out_channels=384,kernel_size=1,stride=1,padding=0)
        self.outConv3 = nn.Conv2d(in_channels=384, out_channels=384,kernel_size=1,stride=1,padding=0)
        self.outConv4 = nn.Conv2d(in_channels=256, out_channels=out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        out = []
        x = self.bn1(self.conv(x))
        x = self.pool(self.residualCov1(x))
        jump_x = x
        x = self.residualCov4(self.residualCov3(self.residualCov2(x)))
        x = self.conv2(self.conv1(self.hourglass1(x)))
        jump_x = torch.cat((x, jump_x), dim=1)
        x = self.outConv1(x)
        out.append(x)
        x = self.outConv3(jump_x) + self.outConv2(x)
        x = self.hourglass2(x)
        x = self.conv4(self.conv3(x))
        x = self.outConv4(x)
        out.append(x)
        return out

# 数据集
class StackHourglassDataset(nn.Module):
    def __init__(self, folder):
        super(StackHourglassDataset, self).__init__()
        self.folder = folder
        # 读取文件夹下的所有以.json结尾的文件
        self.data = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.json')]
        self.img = []
        self.label = []
        for json_file in self.data:
            with open(json_file,'r',encoding='utf-8') as file:
                data = json.load(file)
            jpg_file = json_file.replace('.json','.JPG')
            image_label = data['shapes']
            target_points = [0] * 18
            label_dict = {'c2_down':0,
                        'c7_down':1,
                        't5_up':2,
                        't10_up':3,
                        't12_down':4,
                        'l1_up':5,
                        'l2_down':6,
                        's1_up':7,
                        'cf_back':16,
                        'cf_front':17,
                        'cf':8
                        }
            for label in image_label:
                # 向target_points的label_dict[label['label']]位置插入一个点
                if label['label'] == 'cf_back' or label['label'] == 'cf_front':
                    target_points[label_dict[label['label']]] = label['points'][0]
                else:
                    target_points[label_dict[label['label']] * 2] = label['points'][0]
                    target_points[label_dict[label['label']] * 2 + 1] = label['points'][1]


            image = cv2.imread(jpg_file)

            target = []
            for i in range(0, len(target_points) - 2):
                x, y = target_points[i]
                target.append(generate_gaussian_heatmap((image.shape[0], image.shape[1]),(int(x),int(y)),10))

            # 读取图片
            # 直方图均衡化
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(image)
            # 图片宽和高
            h, w = image.shape
            h_factor = h / 1024
            # resize
            image = cv2.resize(cl1, (int(w / h_factor), 1024),interpolation=cv2.INTER_LINEAR)
            # 图像两边补零让宽度能够被64整除
            final_w = int(w / h_factor) + (64 - int(w / h_factor) % 64)
            left = int((final_w - int(w / h_factor)) / 2)
            right = final_w - left - int(w / h_factor)
            image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
            # 将图片转换为tensor
            image = Image.fromarray(image)
            image = torchvision.transforms.ToTensor()(image)
            heatmaps = []
            for heatmap in target:
                heatmap = resize_heatmap(heatmap, (1024, int(w / h_factor)))
                heatmap = cv2.copyMakeBorder(heatmap.numpy(), 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
                heatmap = resize_heatmap(torch.from_numpy(heatmap), (1024//4, final_w//4))
                heatmaps.append(heatmap)
            # 将image转换为tensor
            # 将图片转换为tensor
            image = image.float()
            # 将target转换为tensor
            heatmaps = torch.stack(heatmaps, dim=0)
            self.img.append(image)
            self.label.append(heatmaps)
        print('data loaded')    
    


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]



def generate_gaussian_heatmap(size, center, sigma):
    """
    生成高斯热力图
    :param size: 热力图的大小 (height, width)
    :param center: 高斯分布的中心 (x, y)
    :param sigma: 高斯分布的标准差
    :return: 生成的高斯热力图
    """
    x = torch.arange(0, size[1], 1, dtype=torch.float32)
    y = torch.arange(0, size[0], 1, dtype=torch.float32)
    y = y[:, None]
    
    x0, y0 = center
    heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return heatmap

def resize_heatmap(heatmap, new_size):
    """
    调整高斯热力图的大小
    :param heatmap: 原始高斯热力图 (PyTorch 张量)
    :param new_size: 新的大小 (height, width)
    :return: 调整大小后的高斯热力图 (PyTorch 张量)
    """
    # 将 PyTorch 张量转换为 NumPy 数组
    heatmap_np = heatmap.numpy()
    
    # 使用 OpenCV 调整大小
    resized_heatmap_np = cv2.resize(heatmap_np, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # 将 NumPy 数组转换回 PyTorch 张量
    resized_heatmap = torch.from_numpy(resized_heatmap_np)
    
    return resized_heatmap