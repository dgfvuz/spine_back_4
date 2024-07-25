from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms.functional
from .config import *

def classification_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss


class VR(nn.Module):
    def __init__(self):
        super(VR, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*8*8, 5)
        

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.bn2(x)
        x = x.view(-1, 128*8*8)
        x = self.fc1(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, target_folder, label_file, transform=None):
        self.target_folder = target_folder
        self.files = os.listdir(target_folder)
        self.transform = transform
        df = pd.read_excel(label_file)
        dict = {}
        for i in range(len(df)):
            dict[df.iloc[i, 0]] = torch.tensor(df.iloc[i, 1])
        self.labels = dict

    def __getitem__(self, index):
        data_path = f"{self.target_folder}/{self.files[index]}"
        x = torchvision.io.read_image(data_path)
        y = self.labels[self.files[index]]
        # 将y变成Long
        y = y.long()
        if self.transform:
            x = self.transform(x)
        # 将x变成float
        x = x.float()
        return x, y

    def __len__(self):
        return len(self.files)
    
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch} loss: {loss.item()} total_loss: {total_loss}')

def start_train():
    data_path = './model_runtime/data'
    label_path = './model_runtime/label/label.xlsx'
    model_path = './model_runtime/model/VR_model.pth'
    if not os.path.exists(data_path):
        print('data folder not found')
        exit()
    if not os.path.exists(label_path):
        print('label file not found')
        exit()
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300,400,500,600], gamma=0.2)
    transfrom = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128))
    ])
    dataset = MyDataset(data_path, label_path, transform=transfrom)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for epoch in range(100000):
        train(model, train_loader, optimizer, criterion, device, epoch)
        if (epoch + 1) % 100 == 0:
            print('save model')
            torch.save(model.state_dict(), model_path)
        sheduler.step()
    torch.save(model.state_dict(), model_path)

def load_model(model_path = VR_model,model = None):
    if model == None:
        model = VR()
    if os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    

def VR_eval(image):
    model_path = VR_model
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # x = torchvision.io.read_image(data_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transposed = np.transpose(image_rgb, (2, 0, 1))
    x = torch.from_numpy(image_transposed).type(torch.uint8)
    x = torchvision.transforms.Resize((128, 128))(x)
    x = x.float()
    x = x.unsqueeze(0)
    x = x.to(device)
    y = model(x)
    if device == torch.device('cuda'):
        y = y.cpu()
    # 取y最大下标
    y = y.detach().numpy()
    y = y.argmax()
    return y


if __name__ == '__main__':
    # start_train()
    # eval()
    print("VR.py")