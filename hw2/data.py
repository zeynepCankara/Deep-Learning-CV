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




class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        # define the data
        self.data = []

        # read the data based on the mode
        if self.mode == 'train' or self.mode == 'val':
            self.data_dir = args.data_dir
            self.data_dir = os.path.join(self.data_dir, self.mode)
            self.img_dir = os.path.join(self.data_dir, 'img')
            self.seg_dir = os.path.join(self.data_dir, 'seg')

            train_img_path_list = sorted([file for file in os.listdir(self.img_dir) if file.endswith('.png')])
            train_seg_path_list = sorted([file for file in os.listdir(self.seg_dir) if file.endswith('.png')])


            for i, train_img_path in enumerate(train_img_path_list):
                file_name = os.path.join(self.img_dir, train_img_path)
                self.data.append([file_name, os.path.join(self.seg_dir, train_seg_path_list[i])])
        else:
            # testing
            train_img_path_list = sorted([file for file in os.listdir(args.input_dir) if file.endswith('.png')])

            self.img_dir = args.input_dir

            for i, train_img_path in enumerate(train_img_path_list):
                file_name = os.path.join(self.img_dir, train_img_path)
                self.data.append([file_name, None])

        self.transform = transforms.Compose([
                            transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                            transforms.Normalize(MEAN, STD)
                            ])



    def my_segmentation_transforms(self, image, segmentation):
        if random.random() > 0.5:
            # flip the image
            image = TF.hflip(image)
            segmentation = TF.hflip(segmentation)
        if random.random() > 0.25:
            # rotation
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
        if random.random() > 0.5:
            # rotation
            angle = random.randint(-10, 10)
            image = TF.rotate(image, angle)
            segmentation = TF.rotate(segmentation, angle)
            
        # more transforms ...
        return image, segmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, cls = self.data[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        seg_img = None # case of test

        if cls != None: # training or validation set
            seg_img = Image.open(cls)
        
        if self.mode == 'train':
            img, seg_img = self.my_segmentation_transforms(img, seg_img)
       
        return self.transform(img), torch.Tensor(np.array(seg_img)).long()
       
