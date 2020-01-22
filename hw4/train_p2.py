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
import pickle
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
    feature_size = 2048 * 2
    loss_train = []
    val_acc_cache = []

   
    with open("./data/train_x4.pkl", "rb") as data:
        train_x = pickle.load(data)

    with open("./data/valid_x4.pkl", "rb") as data:
        valid_x = pickle.load(data)
        
    with open("./data/train_labels4.pkl", "rb") as data:
        train_y = pickle.load(data)      

    with open("./data/valid_labels4.pkl", "rb") as data:
        valid_y = pickle.load(data)
   
 
    model = RNN_P2(feature_size)

   
    model = model.cuda()
    model.train() 
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batch_size
    loss = nn.CrossEntropyLoss()
    max_acc  = float('-inf')


    for epoch in range(args.num_epochs):
        cross_entropy_loss = 0
        dataset_len = len(train_x)
    
        perm_idx = np.random.permutation(len(train_x))
        x_shuffle = [train_x[i] for i in perm_idx]
        y_shuffle = np.array(train_y)[perm_idx]
   
        for idx in range(0,dataset_len ,batch_size):
            if dataset_len <= idx+batch_size:
                break
            
            adam.zero_grad()
            x = x_shuffle[idx:idx + batch_size]
            y = y_shuffle[idx:idx + batch_size]
            
      
            x, y, length = pad_batch(x, y, mode='train')
        
            output, _ = model(x.cuda(), length)
            cur_loss = loss(output, y.cuda())
            cur_loss.backward()
            adam.step()
            cross_entropy_loss += cur_loss.cpu().data.numpy()
            print('Current Loss: ', cross_entropy_loss)
        loss_train.append(cross_entropy_loss)
        # validation
        acc_cache = []
        with torch.no_grad():
            model.eval()
            for i in range(len(valid_y)):
                input_valid_X, input_valid_y, valid_lengths = pad_batch([valid_x[i]], [valid_y[i]], mode='test')
                output, _ = model(input_valid_X.cuda(),valid_lengths)
                output_label = torch.argmax(output,1).cpu().data
                acc_cache += (output_label == input_valid_y).numpy()
            acc  = np.mean(acc_cache)
            print("Epoch:", epoch+1, "   Accuracy: ",acc , "     Loss training",cross_entropy_loss)
            val_acc_cache.append(acc)
        if acc  > max_accuracy:
            torch.save(model.state_dict(), "./models/rnn_classification.pkt")
            max_accuracy = acc 
        model.train()



    
        



    