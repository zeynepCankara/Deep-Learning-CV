"""
This model defined based on the something else
"""
import torch
import torch.nn as nn
import torchvision.models as models


import numpy as np
import torchvision.models as models

class SimpleBaselineModel(nn.Module):

    def __init__(self, args):
        super(SimpleBaselineModel, self).__init__()  

        '''Create the baseline model'''
        # define resnet18 with imagenet weights
        
        # img -> (Nx3x352x448)
        self.resnet18 = models.resnet18(pretrained = True)
        self.resnet18 = nn.Sequential(*(list(self.resnet18.children())[:-2]))
        
       
        train_params = args.train_params
        k = args.unfreeze_k
        if train_params == 1:
            # freeze k weights
            cnt = 0
            for params in self.resnet18.parameters():
                if cnt < k:
                    params.requires_grad = False
                else:
                    params.requires_grad = True
                cnt += 1
        elif train_params == 0:
            for params in self.resnet18.parameters():   
                params.requires_grad = False
        else:
            # un-freeze the weights
            pass

        # map -> (Nx512x11x14)
        
        # first block  
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256, 
                                         kernel_size=4, 
                                         stride=2, padding=1, bias=False) 
        self.relu1 = nn.ReLU(inplace=True)
        # map -> (Nx256x22x28)
        
        # second block 
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=256,
                                 out_channels=128, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # map -> (Nx128x44x56)
        
        # third block 
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=128,
                                 out_channels=64, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        # map -> (Nx64x88x112)
        
        
        # forth block
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=64,
                         out_channels=32, 
                         kernel_size=4, 
                         stride=2, padding=1, bias=False)
        
        self.relu4 = nn.ReLU(inplace=True)
        # map -> (Nx32x176x224)

        # fifth block
        self.conv_transpose5 = nn.ConvTranspose2d(in_channels=32,
                         out_channels=16, 
                         kernel_size=4, 
                         stride=2, padding=1, bias=False)
        
        self.relu5 = nn.ReLU(inplace=True)
        # map -> (Nx16x352x448)
        
        # final block
        self.conv1 = nn.Conv2d(in_channels=16,
                         out_channels=9, 
                         kernel_size=1, 
                         stride=1, padding=0, bias=True)
        # map -> (Nx9x352x448)
        
     
    def forward(self, img):
        x = self.resnet18(img)

        x = self.relu1(self.conv_transpose1(x))
        
        x = self.relu2(self.conv_transpose2(x))
        
        x = self.relu3(self.conv_transpose3(x))
        
        x = self.relu4(self.conv_transpose4(x))

        x = self.relu5(self.conv_transpose5(x))
        
        x = self.conv1(x)
    
        return x

class BaselineModel(nn.Module):

    def __init__(self, args):
        super(BaselineModel, self).__init__()  

        '''Create the baseline model'''
        # define resnet18 with imagenet weights
        
        # img -> (Nx3x352x448)
        self.resnet34 = models.resnet34(pretrained = True)
        self.resnet34 = nn.Sequential(*(list(self.resnet34.children())[:-2]))
        train_params = args.train_params
        # deactivate parameters for the improved model
        if train_params == 0:
            for params in self.resnet34.parameters():   
                params.requires_grad = False
        else:
            pass

        # first block  
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256, 
                                         kernel_size=4, 
                                         stride=2, padding=1, bias=False) 
        self.relu1 = nn.ReLU(inplace=True)
        # map -> (Nx256x22x28)
        
        # second block 
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=256,
                                 out_channels=128, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # map -> (Nx128x44x56)
        
        # third block 
        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=128,
                                 out_channels=64, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        # map -> (Nx64x88x112)
        
        
        # forth block
        self.conv_transpose4 = nn.ConvTranspose2d(in_channels=64,
                         out_channels=32, 
                         kernel_size=4, 
                         stride=2, padding=1, bias=False)
        
        self.relu4 = nn.ReLU(inplace=True)
        # map -> (Nx32x176x224)
        

        # fifth block
        self.conv_transpose5 = nn.ConvTranspose2d(in_channels=32,
                         out_channels=16, 
                         kernel_size=4, 
                         stride=2, padding=1, bias=False)
        
        self.relu5 = nn.ReLU(inplace=True)
        # map -> (Nx16x352x448)
        

        # final block
        self.conv1 = nn.Conv2d(in_channels=16,
                         out_channels=9, 
                         kernel_size=1, 
                         stride=1, padding=0, bias=True)
        # map -> (Nx9x352x448)

     
    def forward(self, img):
        x = self.resnet34(img)

        x = self.relu1(self.conv_transpose1(x))

        x = self.relu2(self.conv_transpose2(x))
        
        x = self.relu3(self.conv_transpose3(x))
      
        
        x = self.relu4(self.conv_transpose4(x))
  

        x = self.relu5(self.conv_transpose5(x))

        
        x = self.conv1(x)
    
        return x

 

