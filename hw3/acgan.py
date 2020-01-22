"""
This module contains ACGAN model
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class Generator_ACGAN(nn.Module):
    def __init__(self, image_size = 64):
        super(Generator_ACGAN, self).__init__()
        self.g = nn.Sequential(
            # Block 1 
            nn.ConvTranspose2d( 101, image_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True),

            # Block 2
            nn.ConvTranspose2d(image_size * 8, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True),

            # Block 3
            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True),

            # Block 4 
            nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),

            # Block 5
            nn.ConvTranspose2d(image_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
            
    def forward(self, X):
        b = 0.5
        out_g = self.g(X)/2.0 + b
        return out_g
    
class Discriminator_ACGAN(nn.Module):
    def __init__(self, image_size = 64):
        super(Discriminator_ACGAN, self).__init__()
        self.d = nn.Sequential(
            # Block 1
            nn.Conv2d(3, image_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4
            nn.Conv2d(image_size * 4, image_size * 8, 4, 2, 1),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 5
            nn.Conv2d(image_size * 8, image_size, 4, 1, 0),
        )
        self.distinguisher = nn.Linear(image_size, 1)
        
        self.fc_auxiliary = nn.Linear(image_size, 1) # 1 class
        
        # Output layers of the ACGAN

        # for the class type
        self.softmax = nn.Softmax()

        # for detecting real or fake
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        d_out = self.d(X)

        d_out_1d = d_out.view(-1,64)

        distinguisher  = self.distinguisher(d_out_1d)

        fc_auxiliary = self.fc_auxiliary(d_out_1d)
        
        true_val = self.sigmoid(distinguisher)
        label = self.sigmoid(fc_auxiliary)
        
        return true_val, label