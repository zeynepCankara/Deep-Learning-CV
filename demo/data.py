import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, 'imgs')

        ''' read the data list '''
        json_path = os.path.join(self.data_dir, mode + '.json')
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        ''' set up image path '''
        for d in self.data:
            d[0] = os.path.join(self.img_dir, d[0])
        
        ''' set up image trainsform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([
                               transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                               transforms.Normalize(MEAN, STD)
                               ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, cls = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), cls
