from datetime import datetime
import gc
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import cv2
from .config import *

class TTS(nn.Module):
    def __init__(self):
        super(TTS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*32*32, 128)
        self.fc2 = nn.Linear(128, 2)
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 64*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, target_folder, label_file):
        self.data = os.listdir(target_folder)
        self.target_folder = target_folder
        df = pd.read_excel(label_file)
        dict = {}
        for i in range(len(df)):
            dict[df.iloc[i, 0]] = torch.tensor([df.iloc[i, 1], df.iloc[i, 2]])
        self.labels = dict

    def __getitem__(self, index):
        data_path = os.path.join(self.target_folder, self.data[index])
        img = Image.open(data_path)
        img = img.convert('L')
        transform = torchvision.transforms.ToTensor()
        x = transform(img).float()
        y = self.labels[int(self.data[index].split('.')[0])]
        # 将y变成float
        y = y.float()
        return x, y

    def __len__(self):
        return len(self.data)
    
def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('{}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(datetime.now(),
            epoch, i * len(data), len(train_loader.dataset),
            100. * i / len(train_loader), loss.item()))
        

def start_train():
    data_path = './data'
    label_path = './label.xlsx'
    model_path = './model.pth'
    if not os.path.exists(data_path):
        print('data folder not found')
        exit()
    if not os.path.exists(label_path):
        print('label file not found')
        exit()
    model = TTS()
    model.train()
    model = load_model(model,model_path=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    sheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300,400,500,600], gamma=0.2)
    dataset = MyDataset(data_path, label_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    for epoch in range(100000):
        train(model, train_loader, optimizer, criterion, device, epoch)
        if (epoch + 1) % 100 == 0:
            print('save model')
            torch.save(model.state_dict(), model_path)
        sheduler.step()
    torch.save(model.state_dict(), model_path)

def load_model(model,model_path = TTS_model):
    if model == None:
        model = TTS()
    if os.path.exists(model_path):
        print('load model')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def TTS_eval(img,model = None):
    # 判断img是否为PIL.Image
    if not isinstance(img, Image.Image):
        # cv2转为PIL.Image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if model == None:
        model = TTS()
        model.eval()
        model = load_model(model)
    # 判断img是否为三通道
    if img.mode == 'RGB':
        img = img.convert('L')
    transform = torchvision.transforms.ToTensor()
    x = transform(img).float()
    x = x.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    model.to(device)
    x = x.to(device)
    y = model(x)
    if device == torch.device('cuda'):
        y = y.cpu()
    y = y.detach().numpy()
    y = y[0]

    return y
    

if __name__ == '__main__':
    start_train()