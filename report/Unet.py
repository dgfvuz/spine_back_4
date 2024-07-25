import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from datetime import datetime
from .config import *
import cv2
import numpy as np

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        dice_coef = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1. - dice_coef

class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.layer1_conv = double_conv2d_bn(1,8)
        self.layer2_conv = double_conv2d_bn(8,16)
        self.layer3_conv = double_conv2d_bn(16,32)
        self.layer4_conv = double_conv2d_bn(32,64)
        self.layer5_conv = double_conv2d_bn(64,32)
        self.layer6_conv = double_conv2d_bn(32,16)
        self.layer7_conv = double_conv2d_bn(16,8)
        self.layer8_conv = nn.Conv2d(8,1,kernel_size=3,
                                     stride=1,padding=1,bias=True)
        
        self.deconv1 = deconv2d_bn(64,32)
        self.deconv2 = deconv2d_bn(32,16)
        self.deconv3 = deconv2d_bn(16,8)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        conv1 = self.layer1_conv(x) # 8
        pool1 = F.max_pool2d(conv1,2)
        
        conv2 = self.layer2_conv(pool1) # 16
        pool2 = F.max_pool2d(conv2,2)
        
        conv3 = self.layer3_conv(pool2)# 32
        pool3 = F.max_pool2d(conv3,2)
        
        conv4 = self.layer4_conv(pool3) # 64
        
        convt1 = self.deconv1(conv4) # 32
        concat1 = torch.cat([convt1,conv3],dim=1) # 64
        conv5 = self.layer5_conv(concat1) # 32
        
        convt2 = self.deconv2(conv5) # 16
        concat2 = torch.cat([convt2,conv2],dim=1) # 32
        conv6 = self.layer6_conv(concat2) # 16
        
        convt3 = self.deconv3(conv6) # 8
        concat3 = torch.cat([convt3,conv1],dim=1) # 16
        conv7 = self.layer7_conv(concat3) # 8
        
        conv8 = self.layer8_conv(conv7) # 1
        outp = self.sigmoid(conv8)
        return outp

class MyDataset(Dataset):
    def __init__(self, data_folder, target_folder,transform = None, target_transform = None):
        self.label_names = os.listdir(target_folder)
        self.data_folder = data_folder
        self.target_folder = target_folder
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.label_names)

    def __getitem__(self, index):
        label_path = os.path.join(self.target_folder,self.label_names[index])
        data_path = os.path.join(self.data_folder,self.label_names[index])
        # 判断系统类型
        if os.name == 'nt':
            label_path = label_path.replace('/','\\')
            data_path = data_path.replace('/','\\')
        else:
            label_path = label_path.replace('\\','/')
            data_path = data_path.replace('\\','/')
        data = torchvision.io.read_image(data_path)
        label = torchvision.io.read_image(label_path)
        # 将label中的值不为0的像素点设置为1
        label[label!=0] = 1
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        # 判断data是否是灰度图
        if data.size(0) == 3:
            data = torchvision.transforms.functional.rgb_to_grayscale(data)
        # 判断label是否是灰度图
        if label.size(0) == 3:
            label = torchvision.transforms.functional.rgb_to_grayscale(label)
        # 将data和label数据类型转化为float
        data = data.float()
        label = label.float()
        return data,label

def start_Unet_train(data_folder='./model_runtime/train/data',target_folder='./model_runtime/train/target',model_path='./model_runtime/model/model.pth'):
    if not os.path.exists(data_folder):
        print('data folder not exists')
        exit()
    if not os.path.exists(target_folder):
        print('target folder not exists')
        exit()
    print('loading dataset')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(contrast=[0.5,2.0],brightness=[0.5,2.0],saturation=[0.5,2.0],hue=[-0.5,0.5])
    ])
    dataset = MyDataset(data_folder,target_folder,transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
    print('dataset size:',len(dataset))
    print('creating model')
    model = Unet()
    # 加载模型
    if os.path.exists(model_path):
        print('loading model')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else :
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    sheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.1)
    # 判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = SoftDiceLoss()
    print('start training')
    for epoch in range(10000000):
        for i,(data,label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion.forward(output.view(-1), label.view(-1))
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('{}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(datetime.now(),
                epoch, i * len(data), len(dataloader.dataset),
                100. * i / len(dataloader), loss.item()))
        sheduler.step()
        if epoch % 100 == 0:
            print('saving model')
            torch.save(model.state_dict(),model_path)
            print('model saved')
    print('training finished')
    print('saving model')
    torch.save(model.state_dict(),model_path)
    print('model saved')


def start_Unet_test(data_folder='./model_runtime/test/data',target_folder='./model_runtime/test/target',model_path='./model_runtime/model/model.pth'):
    if not os.path.exists(data_folder):
        print('data folder not exists')
        exit()
    if not os.path.exists(target_folder):
        print('target folder not exists')
        exit()
    print('loading dataset')
    dataset = MyDataset(data_folder,target_folder)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    print('dataset size:',len(dataset))
    print('creating model')
    model = Unet()
    # 加载模型
    if os.path.exists(model_path):
        print('loading model')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else :
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    # 判断是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 设置为测试模式
    model.eval()
    print('start test')
    success = 0
    fail_list = []
    criterion = SoftDiceLoss()
    with torch.no_grad():
        for i,(data,label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion.forward(output, label)
            print('{}: Test: [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(datetime.now(),
            i * len(data), len(dataloader.dataset),
            100. * i / len(dataloader), loss.item()))
            if loss.item() < 0.0001:
                success += 1
            else:
                fail_list.append(i)
    print('test finished')
    print('success:',success)
    print('success rate',success/len(dataset)*100,'%')
    print('fail:',len(fail_list))


def get_Unet_output(image):
    model = Unet()
    model_path = Unet_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    if not os.path.exists(model_path):
        print('model not exists')
        return None
    model.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
    model.eval()
    model.to(device)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将numpy数组转换为张量
    tensor_image = torch.from_numpy(rgb_image)
    # 调整张量的维度，使颜色通道在前
    tensor_image = tensor_image.permute(2, 0, 1)
    # tensor_image = torchvision.io.read_image(input_path)
    # 判断是否是灰度图
    if tensor_image.size(0) == 3:
        tensor_image = torchvision.transforms.functional.rgb_to_grayscale(tensor_image)
    tensor_image = tensor_image.float().unsqueeze(0)
    tensor_image = tensor_image.to(device)
    output = model(tensor_image)
    if device == torch.device('cuda'):
        output = output.cpu()
    output = output.squeeze()
    output = output.detach().numpy()
    # 将output大于0.8的像素点设置为255，小于0.8的像素点设置为0
    output[output>0.5] = 255
    output[output<=0.5] = 0
    output = output.astype(np.uint8)
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 过滤面积小于1000的轮廓
    contours = [contour for contour in contours if cv2.contourArea(contour) > 3000]
    # 找出位于轮廓范围内的所有像素点
    mask = np.zeros_like(output)
    cv2.drawContours(mask, contours, -1, (255), -1)
    output = cv2.bitwise_and(output, mask)
    return output