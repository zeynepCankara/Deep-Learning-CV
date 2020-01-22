import os
import json
import torch
import scipy.misc
import pickle

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class DATA_P1(Dataset):
    # add args later to constructor
    def __init__(self, mode='train'):
        self.mode =  mode
        if self.mode == "train":
            with open("train_x_p1_high_np.pkl", "rb") as data:
                data = pickle.load(data)
            
                
        if self.mode == "valid":
            with open("valid_x_p1_high_np.pkl", "rb") as data:
                data = pickle.load(data)
        print(len(data))
  

        ''' read the labels from the csv file'''
        if self.mode == 'train':
          label_dir = 'gt_train.csv'
        else:
          label_dir = 'gt_valid.csv'
        self.label_dir = os.path.join('hw4_data/TrimmedVideos/label', label_dir)
        df = pd.read_csv(self.label_dir)
        df = df.sort_values(["Video_name"]).reset_index(drop=True)
        actions = df["Action_labels"].tolist()
        print(len(actions))


        # get both action and img paths togather
        self.dataloader = []
        for idx, video in enumerate(data):
          self.dataloader.append((video, actions[idx]))

        
        ''' image transformations '''

 
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Pad((0,40), fill=0, padding_mode='constant'),
                            transforms.Resize(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize(MEAN, STD)
                            ])



    def __len__(self):
        """ Length of the dataset """
        return len(self.dataloader)

    def __getitem__(self, idx):

        ''' get data '''
        frames, action = self.dataloader[idx]
        torch_frames = []
        for frame in frames:
          torch_frames.append(self.transform(frame))
        return  torch_frames, action