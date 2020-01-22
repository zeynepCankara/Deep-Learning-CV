import os
import csv
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def number_value(path_name):
  num_start = path_name.find('/')
  num_str = path_name[num_start+1:-4]
  num = 0
  cnt = 0
  while len(num_str) > 0:
    num += int(num_str[-1]) * (10**cnt)
    cnt += 1
    num_str = num_str[:-1]
  return num

def number_value_test(path_name):
  num_end = path_name.find('.')
  num_str = path_name[:num_end]
  num = 0
  cnt = 0
  while len(num_str) > 0:
    num += int(num_str[-1]) * (10**cnt)
    cnt += 1
    num_str = num_str[:-1]
  return num


class DATA_DANN_TRAIN(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.mode =  args.mode
        self.load =  args.load

        self.source = args.source
        self.target = args.target

        self.data_dir = args.data_dir
        self.image_size =  args.image_size

        self.img_dir_load = os.path.join(self.data_dir, self.load)
        self.img_dir = os.path.join(self.img_dir_load, self.mode)

        ''' read the CSV data labels '''
        label_dict  = dict()
        if self.mode == 'train':
          self.csv_path = os.path.join(self.img_dir_load, 'train.csv')
        else:
          self.csv_path = os.path.join(self.img_dir_load, 'test.csv')
        with open(self.csv_path, 'r') as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                label_dict[dict(row)['image_name']] = int(dict(row)['label'])
        csvFile.close()

        self.data = []
        ''' set up image path '''
        train_img_path_list = []
        # I needed to divide the dataset into sub directories to make it work in Colab
        if self.mode == 'train':
          if self.load == "mnistm": 
            paths = ['train1', 'train2', 'train3', 'train4', 'train5', 'train6']
          else:
            paths = ['train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8']
          for path in paths:
            train_img_path_list.extend([os.path.join(path, file) for file in os.listdir(os.path.join(self.img_dir, path)) if file.endswith('.png') and len(file) == 9])
            train_img_path_list.sort(key = number_value)
        else:
            # get the test directory
            train_img_path_list.extend([file for file in os.listdir(self.img_dir) if file.endswith('.png') and len(file) == 9])
            train_img_path_list.sort(key = number_value_test)

        for i, train_img_path in enumerate(train_img_path_list):
                file_name = os.path.join(self.img_dir, train_img_path)
                self.data.append([file_name, label_dict[train_img_path[train_img_path.find('/')+1:] ]])
        
        ''' set up image trainsform '''
        self.transform = transforms.Compose([
                            transforms.Resize(self.image_size ),
                            transforms.CenterCrop(self.image_size ),
                            transforms.ToTensor(), 
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path, cls = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), cls, img_path


""" ========== DATALOADER FOR THE BASH SCRIPTS PROBLEM 3 and 4 ==============="""

class DATA_DANN_TEST(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.mode =  args.mode
        self.load =  args.load

        self.source = args.source
        self.target = args.target

        self.image_size =  28

        self.img_dir = args.data_dir


        self.data = []
        ''' set up image path '''
        train_img_path_list = []
        # get the test directory
        train_img_path_list.extend([file for file in os.listdir(self.img_dir) if file.endswith('.png') and len(file) == 9])
        train_img_path_list.sort()

        for _, train_img_path in enumerate(train_img_path_list):
                file_name = os.path.join(self.img_dir, train_img_path)
                self.data.append(file_name)
        
        ''' set up image trainsform '''
        self.transform = transforms.Compose([
                            transforms.Resize(self.image_size ),
                            transforms.CenterCrop(self.image_size ),
                            transforms.ToTensor(), 
                            transforms.Normalize(MEAN, STD)
                            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.data[idx]
        
        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), img_path