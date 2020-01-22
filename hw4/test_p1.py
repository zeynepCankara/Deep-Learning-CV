"""
    This Module contains Testing P1
"""
# system util imports
import os
import csv
import numpy as np
import pandas as pd
# utils
from reader import readShortVideo
from reader import getVideoList

# custom dataset imports
from models_p1 import CNN_P1_UPPER
from models_p1 import CNN_P1 
import parser
 
# system related
import random

# torch related
import torch
import torchvision

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
 
def transform_frames(frames):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Pad((0,40), fill=0, padding_mode='constant'),
                        transforms.Resize(224), 
                        transforms.ToTensor(), 
                        transforms.Normalize(MEAN, STD)
                        ])
    
    torch_frames = []
    for frame in frames:
        torch_frames.append(transform(frame))
    return  torch_frames

def read_input_dir(val_folder, val_labels_dir):
    """
        Reads the videos from input
    """
    # prepare
    video_paths_cache = []

    # keeps all the images
    frames_cache = []

    video_path = val_folder
    categories_cache = sorted(os.listdir(video_path))


    for category in categories_cache:
        dir_content = sorted(os.listdir(os.path.join(video_path, category)))
        path_names = ["-".join(files.split("-")[:5]) for files in dir_content]
        video_paths_cache.extend(path_names)
        for video in dir_content:
            # computations takes long time on my computer
            frames = readShortVideo(video_path, category, video, downsample_factor=12, rescale_factor=0.5)
            buffer = []
            for frame_img in frames:
                buffer.append(frame_img)
            frames_cache.append(buffer)


    # prepare the labels
    label_dir = val_labels_dir
    df = pd.read_csv(label_dir)
    csv_name_cache = df["Video_name"].tolist()
    df = df.sort_values(["Video_name"]).reset_index(drop=True)
    actions = df["Action_labels"].tolist()



    return frames_cache, actions, video_paths_cache, csv_name_cache
    
def evaluate_test_p1(pred_path, label_path): 
    label_dir = label_path
    df = pd.read_csv(label_dir)
    actions = df["Action_labels"].tolist()

    data = pd.read_csv(pred_path, sep="\n", header=None)
    data.columns = ["Action_labels"]

    correct = 0
    for i, item in enumerate(actions):
        if data['Action_labels'][i] == item:
            correct += 1
    acc  = (correct/len(actions)) * 100
    return acc

if __name__=='__main__':
    """
    Configuration parameters for arguments

    args.mode: train
    args.batch_size: 64
    args.num_epochs: 50
    args.lr: 0.0001
    args.random_seed = 999
    args.resume: # pre-trained model path
    args.val_folder
    args.val_labels_dir
    args.output_dir

    args.test_local: False
    """

    # parse the arguments
    args = parser.arg_parse()

    # Set up a Random Seed
    manual_seed = args.random_seed #999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    """ Parse Arguments"""
    epochs = args.num_epochs
    num_batches = args.batch_size
    lr = args.lr

    val_folder = args.val_folder
    val_labels_dir = args.val_labels_dir
    output_dir = args.output_dir
    # test
    """ Set up Training Variables """
    output_size = 2048 * 9 * 7

    # retrieve the dataset
    frames_cache, actions, video_paths_cache, csv_name_cache = read_input_dir(val_folder, val_labels_dir)
    #print("data processing finished")


    cnn = CNN_P1(57600)
    cnn.cuda()
    cnn.eval()

    resnet50_out_valid = []
    with torch.no_grad():
        for i in range(len(frames_cache)):
            frames = frames_cache[i]
            frames = transform_frames(frames)
            frames = torch.stack(frames)
            cnn_out = cnn.resnet50(frames.cuda()).cpu().view(-1, output_size)
            resnet50_out_valid.append(torch.mean(cnn_out,0))
 

    # dataset format fix
    val_x = torch.stack(resnet50_out_valid)
    val_x = val_x.cuda() 
    val_y = torch.LongTensor(actions)


    # load the pre-trained upper part of the deature extractor

    classifier = CNN_P1_UPPER(output_size)
    classifier.cuda()
    classifier.load_state_dict(torch.load(args.resume))
    classifier.eval()
    with torch.no_grad():
        actions_per_frame = classifier(val_x)
        action = torch.argmax(actions_per_frame,1).cpu().data
        action = action.numpy()
        #metric_acc = np.mean((action == val_y.data.numpy()))
    #print("VALIDATION SET ACCURACY: ", metric_acc)
        
    # Output formatting for writing to the csv file
    correct_order = []
    for name in csv_name_cache:
        correct_order.append(video_paths_cache.index(name))
    action = action[correct_order]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'p1_valid.txt'), "w") as csv_file:
        for idx, pred_res in enumerate(action):
            csv_file.write(str(pred_res))
            if idx != len(action)-1:
                csv_file.write("\n")
    #print("TEST FINISHED")

    #acc = evaluate_test_p1(os.path.join(output_dir, "p1_valid.txt"), val_labels_dir)
    #print("Double Acc check: ", acc)



 
    



 