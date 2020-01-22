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
from models_p2 import RNN_P2
import parser
 
# system related
import random

# torch related
import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

# Things to complete
# 1. Process images get CNN features 
# 2. Use the processed features and convert RNN feedable form
# 3. Save predictions with the assumption val data 769 lines and test data 729 lines

def pad_batch(x, y, mode = 'test'):
    """
    Converts images into RNN feedable format
    """
    if mode == 'test':
        seq = nn.utils.rnn.pad_sequence(x)
        action= torch.LongTensor(y)
        seq_len = [len(x[0])]
    else:
        seq_len = [len(X) for X in x]
        # shuffle the dataset
        idx = np.argsort(seq_len)[::-1]

        # sort according to the sequence length
        x = [x[i] for i in idx]
        seq = nn.utils.rnn.pad_sequence(x)
        seq_len = [len(X) for X in x]
        action= torch.LongTensor(np.array(y)[idx])
    return seq, action, seq_len
 
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
            frames = readShortVideo(video_path, category, video, downsample_factor=12, rescale_factor=0.93)
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
    
def evaluate_test_p2(pred_path, label_path): 
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
    output_size = 2048 * 2

    # retrieve the dataset
    frames_cache, actions, video_paths_cache, csv_name_cache = read_input_dir(val_folder, val_labels_dir)
    #print("data reading done")

    # feed the data to the RNN model obtain CNN features
    model = models.resnet50(pretrained=True)
    # Get rid of the last layer
    model = nn.Sequential(*(list(model.children())[:-1]))
    # freeze parameters
    for params in model.parameters():   
        params.requires_grad = False

    model.cuda()
    model.eval()
    resnet50_out_valid = [] 
    counter = 0
    with torch.no_grad():
        for i in range(len(frames_cache)):
            frames = frames_cache[i]
            frames = transform_frames(frames)
            frames = torch.stack(frames)
            feature = model(frames.cuda()).cpu()
            feature = feature.view(-1, output_size)
            resnet50_out_valid.append(feature)

    #print("training instances done")


    # dataset format fix
    val_x = resnet50_out_valid
    val_y = torch.LongTensor(actions)

    #print("Labels obtained")

 
    # Configure RNN
    classifier = RNN_P2(output_size)
    classifier.cuda()
    classifier.load_state_dict(torch.load(args.resume))
    classifier.eval()

    #acc = [] 
    rnn_outputs = []
    with torch.no_grad():
        classifier.eval()
        for i in range(len(val_y)):
            input_x, input_y, seq_len = pad_batch([val_x[i]], [val_y[i]], mode='test')
            frame_action, _ = classifier(input_x.cuda(), seq_len)
            rnn_out = torch.argmax(frame_action,1).cpu().data
            rnn_outputs.append(rnn_out.numpy())
            #acc.append((rnn_out  == input_y).numpy())
        #metric_acc = np.mean(acc)
    #print("VALIDATION SET ACCURACY: ", metric_acc)


    predictions = []
    for pred in rnn_outputs:
        predictions.append(pred[0])
    predictions = np.array(predictions)
        
    # Output formatting for writing to the csv file
    correct_order = []
    for name in csv_name_cache:
        correct_order.append(video_paths_cache.index(name))
    predictions = predictions[correct_order]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'p2_result.txt'), "w") as csv_file:
        for idx, pred_res in enumerate(predictions):
            csv_file.write(str(pred_res))
            if idx != len(predictions)-1:
                csv_file.write("\n")
    #print("TEST FINISHED")

    #acc = evaluate_test_p2(os.path.join(output_dir, "p2_result.txt"), val_labels_dir)
    #print("Double Acc check: ", acc)



 
    



 