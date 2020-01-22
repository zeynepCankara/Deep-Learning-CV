"""
This module for training the ACGAN model
"""
# system related
from __future__ import print_function
import csv
import os
import random
import numpy as np

# deep learning related
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# image related 
from scipy.misc import imsave
from scipy.misc import imread
import torchvision.datasets as dset
import torchvision.transforms as transforms

# custom modules
import parser
import acgan

def save_model(model, save_path):
    """ Saves the model to the specified path"""
    torch.save(model.state_dict(),save_path)

def read_csv_faces():
  ''' read the csv data '''
  smiles = []
  with open("./hw3_data/face/train.csv", 'r') as csvFile:
      reader = csv.DictReader(csvFile)
      for row in reader:
          smiles.append(float(dict(row)['Smiling']))
  csvFile.close()
  return smiles 

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
  
def get_random_idx(data_len):
        return torch.randperm(data_len)

 
def random_generator_input(num_pairs = 10):
    """This function return random generator input """
    # will used to represent smiling
    ones = np.ones(num_pairs)
    # will used to represent non-smiling
    zeros = np.zeros(num_pairs)
    # concatinate for input pairs
    label_tensor = np.hstack((ones,zeros))
    label_tensor= torch.from_numpy(label_tensor).view(20,1,1,1).type(torch.FloatTensor)
    # random noise
    random_tensor = torch.randn(10, 100, 1, 1)
    random_tensor = torch.cat((random_tensor,random_tensor))

    generator_input = Variable(torch.cat((random_tensor, label_tensor),1))

    return generator_input.cuda()

if __name__=='__main__':

    args = parser.arg_parse()
    # set up the seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    """ DATA PRE-PROCESSING """

    # pre-process the numpy array
    img_dir = args.data_dir # './hw3_data_new/face/train/'
    
    ''' set up image path '''
    train_img_path_list = []
    # since the module does not fit into your memory
    paths = ['train1', 'train2', 'train3', 'train4']
    for path in paths:
        train_img_path_list.extend([os.path.join(path, file) for file in os.listdir(os.path.join(img_dir, path)) if file.endswith('.png')])
    # sort the data accorfing 
    train_img_path_list.sort(key = number_value)

    images = []
    #data = []
    for i, train_img_path in enumerate(train_img_path_list):
            file_name = os.path.join(img_dir, train_img_path)
            #print(i) # to keep track of the image processed
            img = imread(file_name) # skimage.io.imread(file_name)
            images.append(img)
            #data.append([file_name, smile_dict[train_img_path[train_img_path.find('/')+1:] ]])

    # convert to the correct format
    images = np.array(images)/255
    images = images.transpose(0,3,1,2)

    #print("Shape of the images", images.shape)

    np.save("face_train.npy", images)

    # prepare the csv labels
    smiles = read_csv_faces()
    smiling_label = np.array(smiles)

    """ MODEL TRAINING """
    """  Set up some training parameters """
    batch_size = args.batch_size # 64
    nz = args.nz # 100
    beta_1 = args.beta1 # 0.5
    beta_2 = args.beta2 # 0.999
    lr = args.lr #0.0002
    epochs = args.num_epochs # 100
    

    # To show some metrics
    discriminator_loss = list()
    generator_loss = list()

    train_X = torch.from_numpy(images).type(torch.FloatTensor)
    label_X = torch.from_numpy(smiling_label).type(torch.FloatTensor).view(-1,1,1,1)
    len_dataset = len(train_X)

    generator_input = random_generator_input(num_pairs = 10)

    # losses
    detector_loss = nn.BCELoss()
    classifier_loss = nn.BCELoss()

    # define models
    G_acgan = acgan.Generator_ACGAN().cuda()
    D_acgan = acgan.Discriminator_ACGAN().cuda()

    # setup optimizer ->->-> Adam optimizers
    G_optimizer = optim.Adam(G_acgan.parameters(), lr=lr , betas=(beta_1,beta_2))
    D_optimizer = optim.Adam(D_acgan.parameters(), lr=lr, betas=(beta_1,beta_2))

    print("======= START TRAINING ============")
    for epoch in range(1, epochs + 1):
        print("Current Epoch >>> === >>> === >>>", epoch)
        current_d_loss = 0.0
        current_g_loss = 0.0

   
        dataset_len = len(train_X)

        # Get a random sample from the data
        random_idx = get_random_idx(len(train_X))
        x = train_X[random_idx]
        label = label_X[random_idx]
        
        # construct training batch
        for idx in range(0, len_dataset, batch_size):
            if dataset_len <= idx + batch_size:
                # prevent going out of the index
                break
               
            # zero the parameter gradients
            D_acgan.zero_grad()
            x_in = x[idx:idx + batch_size]
            label_in = label[idx:idx+batch_size]

            # image = real -> label = real
            img_real = Variable(x_in).cuda()
            class_real = Variable(label_in).cuda()

            label_real = Variable(torch.ones((batch_size))).cuda()
            true_val, label = D_acgan(img_real)

            # Calculate the losses
            d_truth_loss_r = detector_loss(true_val, class_real.view(batch_size,1))
            d_class_loss_r  = classifier_loss(label, label_real.view(batch_size,1))

       

            # img -> fake label
            random_img = torch.randn(batch_size, nz, 1, 1)
            class_f = torch.from_numpy(np.random.randint(2, batch_size)).view(batch_size,1,1,1)
            random_img_vector = Variable(torch.cat((random_img, class_f.type(torch.FloatTensor)),1)).cuda()

            label_f =  Variable(torch.zeros((batch_size))).cuda()
            class_f  = Variable(class_f.type(torch.FloatTensor)).cuda()

            # obtain the fake image
            fake_image = G_acgan(random_img_vector)
            true_val, label = D_acgan(fake_image.detach())
            d_truth_loss_f = detector_loss(true_val, label_f.view(batch_size,1))
            d_class_loss_f  = classifier_loss(label, class_f.view(batch_size,1))
    
            # discriminator update
            D_train_loss = ((d_truth_loss_r + d_class_loss_r)/2) + (np.sum(d_truth_loss_f , d_class_loss_f)/2)
            D_train_loss.backward()
          
            discriminator_epoch_l += (D_train_loss.item())
            D_optimizer.step()
            
            #### train Generator
            repeat = 2
            for k in range(repeat):
                G_acgan.zero_grad()
                # generate fake image
                random_vector = torch.randn(batch_size, nz, 1, 1)
                class_f = torch.from_numpy(np.random.randint(2 , batch_size)).view(batch_size,1,1,1)

                input_vector = Variable(torch.cat((random_vector ,class_f.type(torch.FloatTensor)),1)).cuda()

                class_f = Variable(class_f.type(torch.FloatTensor)).cuda()
                fake_generator_l = Variable(torch.ones((batch_size))).cuda()

                fake_image = G_acgan(input_vector)
                true_val, label  = D_acgan(fake_image)
                g_detect_l = detector_loss(true_val, fake_generator_l.view(batch_size,1))
                g_class_l = classifier_loss(label, class_f.view(batch_size,1))
                generator_train_l = g_detect_l  + g_class_l 
                generator_train_l .backward()
                G_optimizer.step()
            
            generator_epoch_l += (generator_train_l .item())
        print("Discriminator Loss => ",discriminator_epoch_l /(len_dataset))
        print("Generator Loss => ", generator_epoch_l/(len_dataset))
        discriminator_loss.append(discriminator_epoch_l /(len_dataset))
        generator_loss.append(generator_epoch_l/(len_dataset))
        
 
    # save the model
    save_model(G_acgan, "./models/acgan.pkt")
 