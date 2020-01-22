import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms
import torch

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# defined by me
import torchvision.transforms.functional as TF
import random

# ImageNet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]




class DATA_TEST(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        # define the data
        self.data = []


        # testing
        train_img_path_list = sorted([file for file in os.listdir(args.input_dir) if file.endswith('.png')])

        self.img_dir = args.input_dir

        for _, train_img_path in enumerate(train_img_path_list):
            file_name = os.path.join(self.img_dir, train_img_path)
            self.data.append(file_name)

        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path  = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')


        return self.transform(img)