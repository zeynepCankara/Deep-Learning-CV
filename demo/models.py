import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()  

        ''' declare layers used in this network'''
        # first block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # 64x64 -> 64x64
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        # second block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        
        # third block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 16x16 -> 16x16
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

        # classification
        # self.avgpool = nn.AvgPool2d(16)
        # self.fc = nn.Linear(64, 4)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(128, 4)

    def forward(self, img):

        x = self.relu1(self.bn1(self.conv1(img)))
        x = self.maxpool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        x = self.avgpool(x).view(x.size(0),-1)
        x = self.fc(x)

        return x

