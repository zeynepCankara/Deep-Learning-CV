import random
import torch
from PIL import Image
from glob import glob
import numpy as np
import os

def invert_bw(mask):
    mask=np.array(mask).astype(np.uint8)
    mask[mask>200]=255
    mask[mask<=200]=0
    mask=np.invert(mask)
    mask = Image.fromarray(mask)
    return mask

class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform):
        super(Data_load, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.img_root = img_root
        self.mask_root = mask_root
      
        images = sorted(os.listdir(img_root))
        self.paths = [f"{img_root}/{image}" for image in images]

        masks = sorted(os.listdir(mask_root))
        self.mask_paths = [f"{mask_root}/{mask}" for mask in masks]

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))
        # rand_num = random.randint(0, self.N_mask - 1)
        # mask = Image.open(self.mask_paths[rand_num])
        # print("img root, mask root, len(self.mask_paths), index",self.img_root,self.mask_root,len(self.mask_paths), index)
        mask = Image.open(self.mask_paths[index])
        mask = self.mask_transform(invert_bw(mask.convert('RGB')))
        # print("gt_img.shape, mask.shape: ",gt_img.shape, mask.shape)
        return gt_img , mask

    def __len__(self):
        return len(self.paths)

