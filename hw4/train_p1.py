"""
    This Module contains Training Files for the CNN extractor
"""
# system util imports
import os
import numpy as np

# custom dataset imports
from data_p1 import DATA_P1
from models_p1 import CNN_P1
from models_p1 import CNN_P1_UPPER
import parser
from visualizations import plot_p1_train_info, plot_embedding

# system related
import random
import os

# torch related
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

# torch dataset related
from torchvision import datasets
from torchvision import transforms
from sklearn.manifold import TSNE



def extract_features():
    """
        Extracts CNN features using resnet50
    """
    # Load the training dataset
    train = DATA_P1(mode='train')

    # Load the validation dataset
    val = DATA_P1(mode='valid')

    output_size = 2048 * 9 * 7
    cnn = CNN_P1(57600)
    cnn.cuda()
    cnn.eval()
    
    resnet50_out_train = []
    with torch.no_grad():
        for i in range(len(train)):
            frames, _ = train[i]
            frames  = torch.stack(frames)
            cnn_out = cnn.resnet50(frames.cuda()).cpu().view(-1, output_size)
            resnet50_out_train.append(torch.mean(cnn_out ,0))
        

    print("Training dataset processed...")

    resnet50_out_valid = []
    with torch.no_grad():
        for i in range(len(val)):
            frames, _ = val[i]
            frames  = torch.stack(frames)
            cnn_out = cnn.resnet50(frames.cuda()).cpu().view(-1, output_size)
            resnet50_out_valid.append(torch.mean(cnn_out,0))


    print("Validation dataset processed...")
    return resnet50_out_train, resnet50_out_valid, train, val

def process_dataset(resnet50_out_train, resnet50_out_valid, train, val):
    training_y = []
    for i in range(len(train)):
        training_y.append(train[i][1])

    val_y = []
    for i in range(len(val)):
        val_y.append(val[i][1])

    t_x = torch.stack(resnet50_out_train)
    t_y = torch.LongTensor(training_y)

    v_x = torch.stack(resnet50_out_valid)
    v_y = torch.LongTensor(val_y)

    return t_x, t_y, v_x, v_y


if __name__=='__main__':
    """
    Configuration parameters for arguments

    args.mode: train
    args.batch_size: 64
    args.num_epochs: 50
    args.lr: 0.0001
    args.random_seed = 999
    """
    # If you want to visualize the t-sne 
    plot_t_sne = False
    plot_train_loss = False

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
    """ Set up Training Variables """
    output_size = 2048 * 9 * 7
    acc_max = 0.0
    total_loss_t = []
    total_val_acc = []

    # Set the Model
    cnn = CNN_P1_UPPER(output_size).cuda()
    cnn.train()
    # Set the optimizer
    adam = torch.optim.Adam(cnn.parameters(), lr=lr)
    # set the loss
    cross_entropy = nn.CrossEntropyLoss()

    # retrieve dataset
    resnet50_out_train, resnet50_out_valid, train, val = extract_features()
    t_x, t_y, v_x, v_y = process_dataset(resnet50_out_train, resnet50_out_valid, train, val)
    len_t_x = len(t_x)
    for epoch in range(1, epochs + 1):
        print("CURRENT EPOCH =====>>>>   ====>>>> ", epoch)
        epoch_loss = 0.0
        
        # Get a random item from the dataset
        rnd_idx = torch.randperm(len_t_x)
        x = t_x[rnd_idx]
        y = t_y[rnd_idx]

        # Get a batch of sample
        for idx in range(0, len_t_x, num_batches):
            if (idx + num_batches) > len_t_x:
                print("Index out of boundary...")
                break
     
            adam.zero_grad()
            batch_x = x[idx:idx+num_batches].cuda()
            batch_y = y[idx:idx+num_batches].cuda()
   
            # flow gradients
            cnn_out = cnn(batch_x)
            cross_entropy = cross_entropy(cnn_out, batch_y)
            cross_entropy.backward()
            adam.step()

            epoch_loss += np.numpy(cross_entropy.cpu().data)
        print("== TRANING LOSS ==> ==> ==> ", epoch_loss)
        total_loss_t .append(epoch_loss)
        # Check the validation loss
        with torch.no_grad():
            cnn.eval()
            v_x = v_x.cuda()
            out_cnn = cnn(v_x)
            pred_cnn = torch.argmax(out_cnn,1).cpu().data
            acc = np.mean((pred_cnn == v_y).numpy())
            print("== VALIDATION SET ACCURACY ==> ==> ==> ", acc)
            total_val_acc.append(acc)
        if acc > acc_max:
            torch.save(cnn.state_dict(), "./models/cnn_model.pkt")
            acc_max = acc
        cnn.train()
    
    print('Training Finished...')
    if plot_t_sne == True:
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
        print(t_x.detach().cpu().numpy().shape)
        feature_np = t_x.detach().cpu().numpy()
        feature_np = feature_np.reshape(feature_np.shape[0], feature_np.shape[1] )
        print(feature_np.shape)
        #dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
        tsne = tsne.fit_transform(feature_np)
        label_y = t_y.detach().cpu().numpy()
        labels = []
        for i in label_y:
            labels.append(int(i))
        plot_embedding(tsne, labels, 'train', 't-sne')
    if plot_train_loss == True:
        plot_p1_train_info(total_loss_t, total_val_acc, save_dir = "./saved_plot/problem1_loss_acc.png")
    print('Plots Finished...')

    

