
from typing import no_type_check
import os
from matplotlib.colors import Normalize
import numpy as np
from numpy.lib.function_base import select
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
lr = 1e-3
bathc_size = 4
num_epoch = 100

data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'

#cpu ,gpu 디바이스 선택하기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 트랜스폼 구현하기
#numpy 이미지 채널 (y,x,ch)
#tensor 이미지 채널(ch, y,x)
class ToTensor(object):
    def __call__(self,data):
        label, input = data['label'], data['input']

        # numpy -> tensor로 변환하기
        label = label.transpose((2,0,1)).astype(np.float32)
        input = input.transpose((2,0,1)).astype(np.float32)

        # from_numpy -> 텐서플로로 변환할때 사용하는 함수
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        return data

# nomalization 하기
class Nomalization(object):
    def __init__(self, mean = 0.5, std = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        data = {'label' : label, 'input' : input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # numpy 배열 좌우 뒤집기 
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        # numpy배열 상하좌우 반전
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        
        data = {'label': label, 'input': input}
        
        return data


# unet 네트워크 구조

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
            layers = []
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias )]

            layers += [nn.BatchNorm2d(num_features = out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr
    
        #contracting path
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64 )
        self.enc1_1 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels= 128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels= 256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels= 512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size = 2)

        self.enc5_1 = CBR2d(in_channels= 512, out_channels= 1024)

        self.dec5_1 = CBR2d(in_channels=1024 , out_channels= 512)

        # 디코드하기
        self.unpool4 = nn.ConvTranspose2d(in_channels = 512, out_chnnels = 512, kerner_size = 2, stride = 2, padding = 0, bias = True)

        self.dec4_2 = CBR2d(in_channels=1024, out_channels= 512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels= 256)
        self.unpool3 = nn.ConvTranspose2d(in_channels = 256, out_chnnels = 256, kerner_size = 2, stride = 2, padding = 0, bias = True) 

        self.dec3_2 = CBR2d(in_channels=512, out_channels= 256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels= 128)
        self.unpool2 = nn.ConvTranspose2d(in_channels = 128, out_chnnels = 128, kerner_size = 2, stride = 2, padding = 0, bias = True) 

        self.dec2_2 = CBR2d(in_channels=256, out_channels= 128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels= 64)
        self.unpool1 = nn.ConvTranspose2d(in_channels = 64, out_chnnels = 64, kerner_size = 2, stride = 2, padding = 0, bias = True) 

        self.dec1_2 = CBR2d(in_channels=128, out_channels= 64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels= 64)

        self.fc = nn.Conv2d(in_channels = 64, out_channels =2 , kerner_sizze = 1, stride = 1, padding = 0, bias = True)

    #모델 연결하기
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc1_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.enc5_1(enc5_1)

        unpool4 = self.unpool3(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)
        return x



# 데이터 로더 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # 디렉토리에 있는것 가지고 오기
        lst_label = [f for f  in lst_data if f.startswith('label')]
        lst_input = [f for f  in lst_data if f.startswith('input')]
 
        # 라벨 정렬하기
        lst_label.sort()
        lst_label.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self,index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        
        #0~1사이로 변환
        # input = (input - np.min(input))/(np.max(input) - np.min(input)) 

        #input = input - np.min(input)
        #input = input / np.max(input)
        
        label = label / 255.0 - 0.5
        input = input / 255.0 - 0.5

        if label.ndim == 2:
            label = label[:,:,np.newaxis]
        if input.ndim == 2:
            input = input[:,:,np.newaxis]
 
        data = {'input' : input, 'label' : label}

        if self.transform:
            data = self.transform(data)
        
        return data

#transform = transforms.Compose([Nomalization(mean = 0.5, std = 0.5), RandomFlip(), ToTensor()])
transform = transforms.Compose([RandomFlip(), ToTensor()])
dataset_train = Dataset(data_dir= os.path.join(data_dir, 'train'), transform=transform)

data = dataset_train.__getitem__(0) # getitem의 인자값을 바꿀때 dataset 이미지 파일 바뀜
input = data['input']
label = data['label']
print(input.shape)

plt.subplot(121) # 벡터
plt.imshow(input.squeeze())

plt.subplot(122)
plt.imshow(label.squeeze())

plt.show()

