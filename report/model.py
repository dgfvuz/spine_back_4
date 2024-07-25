import gc
import os
import threading
import time
import argparse
import logging
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.append(os.path.split(sys.path[0])[0])
import cv2
import numpy as np
from PIL import Image
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import logging
from PIL import Image
import os.path as osp
from scipy import ndimage
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from .preprocess import preprocess_image,reverse_process
from .calculations import *
from .config import *
lock = threading.Lock()
from memory_profiler import profile

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class DUCK_res(nn.Module):
    def __init__(self, input, output):
        super(DUCK_res, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output),
            nn.Conv2d(output, output, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output),
        )
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(output),
        )
        self.BN = nn.BatchNorm2d(output)

    def forward(self,x):
        x1 = self.conv_x1(x)
        x = self.conv_block(x)
        x = x+x1
        x = self.BN(x)

        return x 
  
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MLP(nn.Module):
    def __init__(self,input,output):
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input,input*4),
            nn.ReLU(),
            nn.Linear(input*4,input*4),
            nn.ReLU(),
            nn.Linear(input*4,output),
            # nn.Softmax()
            nn.BatchNorm2d(output)
        )
    def forward(self,x):
        output = self.model(x)
        return output

class SCI_module(nn.Module):
    def __init__(self, channel, k_size=3):
        super(SCI_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #[b, c, h, w]
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)
 
class duckv2_conv2D_block(nn.Module):
    def __init__(self, input, output):
        super(duckv2_conv2D_block, self).__init__()
        
        self.input_bat = nn.BatchNorm2d(input)
        self.output_bat = nn.BatchNorm2d(output)
        self.wide = widescope_conv2D_block(input, output)
        self.mid = midscope_conv2D_block(input,output)
        self.res_input = DUCK_res(input, output)
        self.res_output = DUCK_res(output, output)
        self.separate = separated_conv2D_block(input,output)
        self.channel_chosen = nn.Sequential(
            nn.Conv2d(output*6, output, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(output),
        )
        self.short_cut = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(output),
        )
        # self.mlp = MLP(output*6, output)
        # self.m = nn.Sequential(
        #     nn.Linear(output*6,output*4),
        #     nn.ReLU(),
        #     nn.Linear(output*4,output*4),
        #     nn.ReLU(),
        #     nn.Linear(output*4,output),
        #     # nn.Softmax()
        #     nn.BatchNorm2d(output)
        # )
        self.sci = SCI_module(output*6,3)
    
    def forward(self, x):
      x = self.input_bat(x)
      x1 = self.wide(x)
      x2 = self.mid(x)
      x3 = self.res_input(x)
      x4 = self.res_input(x)
      x4 = self.res_output(x4)
      x5 = self.res_input(x)
      x5 = self.res_output(x5)
      x5 = self.res_output(x5)
      x6 = self.separate(x)
      sx = self.short_cut(x)
      
      # print(x1.shape, x2.shape, x3.shape, x4.shape,x5.shape,x6.shape)
      # y = x1+x2+x3+x4+x5+x6
      y = torch.cat([x1,x2,x3,x4,x5,x6],dim=1)
      
      # print(y.shape)
      y = self.sci(y)
      # print(y.shape)
      y = self.channel_chosen(y)
      y = y+sx
      y = self.output_bat(y)
      
      return y

class separated_conv2D_block(nn.Module):
    def __init__(self, input, output):
        super(separated_conv2D_block, self).__init__()
        
        self.Conv_block = nn.Sequential(
            nn.Conv2d(input, output, kernel_size = (1, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output),
            nn.Conv2d(output, output, kernel_size = (3,1)),
            nn.ReLU(),
            nn.BatchNorm2d(output),
        )
        
    def forward(self, x):
        return self.Conv_block(x)
    
class midscope_conv2D_block(nn.Module):
    def __init__(self, input, output):
        super(midscope_conv2D_block, self).__init__()
        
        self.Conv_block = nn.Sequential(
          nn.Conv2d(input, output, kernel_size=3, dilation=1, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(output),
          nn.Conv2d(output, output, kernel_size=3, dilation=2, padding=2),
          nn.ReLU(),
          nn.BatchNorm2d(output),
        )
        
    def forward(self, x):
        return self.Conv_block(x)
    
class widescope_conv2D_block(nn.Module):
    def __init__(self, input, output):
        super(widescope_conv2D_block, self).__init__()
        
        self.Conv_block = nn.Sequential(
          nn.Conv2d(input, output, kernel_size=3, dilation=1, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(output),
          nn.Conv2d(output, output, kernel_size=3, dilation=2, padding=2),
          nn.ReLU(),
          nn.BatchNorm2d(output),
          nn.Conv2d(output, output, kernel_size=3, dilation=3, padding=3),
          nn.ReLU(),
          nn.BatchNorm2d(output),
        )
        
    def forward(self, x):
        return self.Conv_block(x)

class Attention_block(nn.Module):
 
    def __init__(self, low, high, output):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(low,
                      output,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), 
            nn.BatchNorm2d(output))
 
        self.W_x = nn.Sequential(
            nn.Conv2d(high,
                      output,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), 
            nn.BatchNorm2d(output))
 
        self.psi = nn.Sequential(
            nn.Conv2d(output, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())
 
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
 
        return x * psi

class DUCKNet(nn.Module):
    def __init__(self, input, output, filters=17):
        super(DUCKNet, self).__init__()
        
        self.t0_down = nn.Conv2d(filters, filters*2, kernel_size=2, stride=2)
        self.down1 = nn.Conv2d(input, filters*2, kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(filters*2, filters*4, kernel_size=2, stride=2)
        self.down3 = nn.Conv2d(filters*4, filters*8, kernel_size=2, stride=2)
        self.down4 = nn.Conv2d(filters*8, filters*16, kernel_size=2, stride=2)
        self.down5 = nn.Conv2d(filters*16, filters*32, kernel_size=2, stride=2)
        
        self.input_duck = duckv2_conv2D_block(input, filters)
        self.s1_duck = duckv2_conv2D_block(filters*2, filters*2)
        self.s2_duck = duckv2_conv2D_block(filters*4, filters*4)
        self.s3_duck = duckv2_conv2D_block(filters*8, filters*8)
        self.s4_duck = duckv2_conv2D_block(filters*16, filters*16)
        self.s5_res = DUCK_res(filters*32, filters*32)
        self.s5_res_2 = DUCK_res(filters*32, filters*16)
        
        # self.up1 = Up(filters*16, filters*16)
        # self.up2 = Up(filters*8, filters*8)
        # self.up3 = Up(filters*4, filters*4)
        # self.up4 = Up(filters*2, filters*2)
        # self.up5 = Up(filters, filters)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.c4_duck = duckv2_conv2D_block(filters*16, filters*8)
        self.c3_duck = duckv2_conv2D_block(filters*8, filters*4)
        self.c2_duck = duckv2_conv2D_block(filters*4, filters*2)
        self.c1_duck = duckv2_conv2D_block(filters*2, filters)
        self.c0_duck = duckv2_conv2D_block(filters, filters)

        self.att5 = Attention_block(filters*16,filters*16,filters*16)
        self.att4 = Attention_block(filters*8,filters*8,filters*8)
        self.att3 = Attention_block(filters*4,filters*4,filters*4)
        self.att2 = Attention_block(filters*2,filters*2,filters*2)

        self.channel_chosen5 = nn.Sequential(
            nn.Conv2d(filters*32, filters*16, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters*16),
        )
        self.channel_chosen4 = nn.Sequential(
            nn.Conv2d(filters*16, filters*8, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters*8),
        )
        self.channel_chosen3 = nn.Sequential(
            nn.Conv2d(filters*8, filters*4, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters*4),
        )
        self.channel_chosen2 = nn.Sequential(
            nn.Conv2d(filters*4, filters*2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(filters*2),
        )
        
        self.output_duck = nn.Conv2d(filters, output, kernel_size=1)
        
    def forward(self, x):
        t0 = self.input_duck(x)
        p1 = self.down1(x)
        p2 = self.down2(p1)
        p3 = self.down3(p2)
        p4 = self.down4(p3)
        p5 = self.down5(p4)
        # print("t0,p1,p2,p3,p4,p5:",t0.shape, p1.shape, p2.shape, p3.shape, p4.shape, p5.shape )
        
        l1i = self.t0_down(t0)
        s1 = l1i + p1
        t1 = self.s1_duck(s1)
        # print("l1i,s1,t1:",l1i.shape, s1.shape, t1.shape)
        
        l2i = self.down2(t1)
        s2 = p2+l2i
        t2 = self.s2_duck(s2)
        # print("l2i,s2,t2:",l2i.shape, s2.shape, t2.shape)
        
        l3i = self.down3(t2)
        s3 = p3+l3i
        t3 = self.s3_duck(s3)
        # print("l3i,s3,t3:",l3i.shape, s3.shape, t3.shape)
        
        l4i = self.down4(t3)
        s4 = p4+l4i
        t4 = self.s4_duck(s4)
        # print("l14,s4,t4:",l4i.shape, s4.shape, t4.shape)
        
        l5i = self.down5(t4)
        s5 = p5+l5i
        t51 = self.s5_res(s5)
        t53 = self.s5_res_2(t51)
        # print("l5i,s5,t51,t53:",l5i.shape, s5.shape, t51.shape, t53.shape)
        
        l5o = self.up(t53)
        att5 = self.att5(g=l5o,x=t4)
        l5o = torch.cat((att5,l5o),dim=1)
        c4 = self.channel_chosen5(l5o)
        q4 = self.c4_duck(c4)
        # print("l5o,c4,q4:",l5o.shape, c4.shape, q4.shape)
        
        l4o = self.up(q4)
        att4 = self.att4(g=l4o,x=t3)
        l4o = torch.cat((att4,l4o),dim=1)
        c3 = self.channel_chosen4(l4o)
        q3 = self.c3_duck(c3)
        # print("l4o,c3,q3:",l4o.shape, c3.shape, q3.shape)
        
        l3o = self.up(q3)
        att3 = self.att3(g=l3o,x=t2)
        l3o = torch.cat((att3,l3o),dim=1)
        c2 = self.channel_chosen3(l3o)
        q2 = self.c2_duck(c2)
        # print("l3o,c2,q2:",l3o.shape, c2.shape, q2.shape)
        
        l2o = self.up(q2)
        att2 = self.att2(g=l2o,x=t1)
        l2o = torch.cat((att2,l2o),dim=1)
        c1 = self.channel_chosen2(l2o)
        q1 = self.c1_duck(c1)
        # print("l2o,c1,q1:",l2o.shape, c1.shape, q1.shape)
        
        l1o = self.up(q1)
        c0 = t0+l1o
        z1 = self.c0_duck(c0)
        # print("l1o,c0,z1:",l1o.shape, c0.shape, z1.shape)
        
        output = self.output_duck(z1)
        # print("output:",output.shape)
        
        return output
        
class SingleDataset(Dataset):
    def __init__(self, image_list, img_file, scale=1):
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = image_list
        self.files = []

        for name in self.ids:
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        datafiles = self.files[i]
        name = datafiles["name"]
        img = Image.open(datafiles["img"]).convert('RGB')
        # origin
        img = img.resize((512,512))
        # mask_size = mask.size
        mask_size = img.size
        img = self.preprocess(img, self.scale)
        return {
            'image1': torch.from_numpy(img).type(torch.FloatTensor),
            'mask_size': mask_size,
            'name': name
        }

class BasicDataset(Dataset):
    def __init__(self, image_list, imgs_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [i_id.strip() for i_id in open(image_list,encoding='utf-8')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.files = []

        for name in self.ids:
            img_file = osp.join(imgs_dir, name)

            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        datafiles = self.files[i]
        name = datafiles["name"]
        img = Image.open(datafiles["img"]).convert('RGB')
        
        # origin
        img = img.resize((512,512))

        # mask_size = mask.size
        mask_size = img.size

        img = self.preprocess(img, self.scale)

        return {
            'image1': torch.from_numpy(img).type(torch.FloatTensor),
            'mask_size': mask_size,
            'name': name
        }


MASK_THRESHOLD = 0.5
# path是图片的路径
# 输入为一个512*512的图片
# 输出为一个512*512的mask,Image格式
def eval(path):
    print('evaluating', path)
    net = DUCKNet(3,1,32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # 这是模型的路径
    model_path = DuckNet_model

    img_list = [path.split('/')[-1]]

    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded !")

    net.eval()

    test_dataset = SingleDataset(img_list, path)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs = batch['image1']
            size = batch['mask_size']
            name = batch['name']
            # mask = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            # 单
            masks_pred = net(imgs)

            masks_pred = torch.sigmoid(masks_pred)    #for AU

            masks_pred = masks_pred.squeeze(0)

            size1 = size[0].cpu().numpy()
            size2 = size[1].cpu().numpy()
            size = [size2[0], size1[0]]

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(size),
                    transforms.ToTensor()
                ]
            )

            masks_pred = tf(masks_pred.cpu())
            masks_pred = masks_pred.squeeze().cpu().numpy()
            masks_pred = masks_pred > MASK_THRESHOLD

            # 填补空洞
            masks_pred = ndimage.binary_fill_holes(masks_pred).astype(int)

            masks_pred = Image.fromarray((masks_pred * 255).astype(np.uint8))
            # masks_pred.save('./model_runtime/result/' + os.path.basename(name[0]), 'png')
    return masks_pred

if __name__ == "__main__": 
    path = './model_runtime/data/曾裕轩.jpg'
    eval(path)



#  仅仅只有一种方法: 需要把结果恢复到原图像尺寸
def getResult(image_file):
    lock.acquire()
    try:
        print("开始处理图片........")
        raw_image = open_img(image_file.name)
        path, preprocess_detail = preprocess_image(image_file.name)
        res = eval(path)

        res = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
        res = reverse_process(res, preprocess_detail)
        # 转为cv2
        # 计算骶骨倾斜
        SCRL_res,SCRL_points = find_SCRL(res)
        # 计算胸1锥体倾斜角
        T1_tile_angle_res,T1_tile_angle_points = find_T1_tile_angle(res)

        CSVL_res,CSVL_points = draw_CSVL(res)

        C7PL_res,C7PL_points = draw_C7PL(res)

        avt_res, avt_points = find_avt(res)

        coronal_balance_res, coronal_balance_points = find_coronal_balance(res)

        cobb_res, cobb_points, bone_box = find_Cobb_new(res)

        rsh_res, rsh_point = find_rsh(raw_image)

        angle_res, angle_point = find_clavicle_angle(raw_image)

        tts_res, tts_point = find_TTS(raw_image,res)

        pixels_per_centimeter = find_centermeter_per_pixel(raw_image)
        # 创建json对象
        result = {
        "pixes_per_centimeter": pixels_per_centimeter,
        "bone_box":bone_box,
        "颈胸弯":{
            "result": "正常",
            "cobb":0,
            "上端椎":"",
            "顶椎":"",
            "下端椎":"",
            "Nash-Moe旋转":"",
            "points": []
        },
        "上胸弯":{
            "result": "正常",
            "cobb":0,
            "上端椎":"",
            "顶椎":"",
            "下端椎":"",
            "Nash-Moe旋转":"",
            "points": []
        },
        "胸弯":{
            "result": "正常",
            "cobb":0,
            "上端椎":"",
            "顶椎":"",
            "下端椎":"",
            "Nash-Moe旋转":"",
            "points": []       
        },
        "胸弯2":{
            "result": "正常",
            "cobb":0,
            "上端椎":"",
            "顶椎":"",
            "下端椎":"",
            "Nash-Moe旋转":"",
            "points": []
        },
        "胸腰弯":{
            "result": "正常",
            "cobb":0,
            "上端椎":"",
            "顶椎":"",
            "下端椎":"",
            "Nash-Moe旋转":"",
            "points": []
        },
        "腰弯":{
            "result": "正常",
            "cobb":0,
            "上端椎":"",
            "顶椎":"",
            "下端椎":"",
            "Nash-Moe旋转":"",
            "points": []
        },
        "冠状面平衡":{
            "result": coronal_balance_res,
            "points": coronal_balance_points
        },
        "锁骨角":{
            "result": angle_res,
            "points": angle_point
        },
        "csvl":{
            "result": CSVL_res,
            "points": CSVL_points
        },
        "c7pl":{
            "result": C7PL_res,
            "points": C7PL_points
        },
        "顶椎偏距":{
            "result": avt_res,
            "points": avt_points
        },
        "胸廓躯干倾斜":{
            "result": tts_res,
            "points": tts_point
        },
        "影像学肩高度":{
            "result": rsh_res,
            "points": rsh_point
        },
        "胸1锥体倾斜角":{
            "result": T1_tile_angle_res,
            "points": T1_tile_angle_points
        },
        "骶骨倾斜角":{
            "result": SCRL_res,
            "points": SCRL_points
        },
    }
        for i in range(len(cobb_res)):
            result[cobb_res[i][0]]['result'] = cobb_res[i][1]
            result[cobb_res[i][0]]['上端椎'] = cobb_res[i][2]
            result[cobb_res[i][0]]['下端椎'] = cobb_res[i][3]
            result[cobb_res[i][0]]['顶椎'] = cobb_res[i][4]
            result[cobb_res[i][0]]['Nash-Moe旋转'] = cobb_res[i][5]
            result[cobb_res[i][0]]['cobb'] = cobb_res[i][6]
            result[cobb_res[i][0]]['points'] = cobb_points[i]
        print("处理完成........")
    finally:
        lock.release()
    return result
