# system related
from __future__ import print_function
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

# image related 
from scipy.misc import imsave
import torchvision.datasets as dset
import torchvision.transforms as transforms

# custom modules
import parser
import generator
import discriminator

def save_model(model, save_path):
    """ Saves the model to the specified path"""
    torch.save(model.state_dict(),save_path)

 
if __name__=='__main__':

    args = parser.arg_parse()
    # set up the seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # get the dataset
    dataset = dset.ImageFolder(root=args.data_dir,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")


    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # =============== GENEREATOR =============>>>
    n_gen = generator.Generator(args.ngpu).to(device)
    
 
    if (device.type == 'cuda') and (args.ngpu > 1):
        n_gen = nn.DataParallel(n_gen, list(range(args.ngpu)))

 
    n_gen.apply(weights_init)

    # ============ DISCRIMINATOR =====>>>
    n_disc= discriminator.Discriminator(args.ngpu).to(device)

  
    if (device.type == 'cuda') and (args.ngpu > 1):
        n_disc = nn.DataParallel(n_disc, list(range(args.ngpu)))
 
    n_disc.apply(weights_init)

    # ====== LOSS ==========>>>
    loss = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # ============== OPTIMISERS ===============>>>
    optimizerD = optim.Adam(n_disc.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(n_gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # ============== TRANING LOOP ==============>>>

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
 
    print("Starting Training...")
    # For each epoch
    for epoch in range(args.num_epochs):
        # For each batch  
        for i, data in enumerate(dataloader, 0):

            ############################
            # Update Discriminator network 
            # Goal: Maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
        
            n_disc.zero_grad()
          
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
           
            output = n_disc(real_cpu).view(-1)
           
            errD_real = loss(output, label)
         
            errD_real.backward()
            D_x = output.mean().item()

 
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
             
            fake = n_gen(noise)
            label.fill_(fake_label)
             
            output = n_disc(fake.detach()).view(-1)
             
            errD_fake = loss(output, label)
             
            errD_fake.backward()
            D_G_z1 = output.mean().item()
             
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # Updating Generator network 
            # GOAL: Maximize log(D(G(z)))
            ###########################
            n_gen.zero_grad()
            label.fill_(real_label)   
            output = n_disc(fake).view(-1)
            # Calculate G's loss based on this output
            errG = loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Training Statistics
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Saving Generator and Discriminator losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1



    # save the model Generator and Discriminator
    save_model(n_disc,  os.path.join(args.save_dir_model, 'model_discriminator.pth.tar'))
    save_model(n_gen, os.path.join(args.save_dir_model, 'model_generator.pth.tar'))


    # Get the real batch from the dataloader
    real_batch = next(iter(dataloader))

    # saving the real image
    img = np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=5, normalize=True).cpu(),(1,2,0))
    imsave(os.path.join(args.out_dir_p1_p2, "real_img.png"), img)

    # Save the fake image
    img = np.transpose(img_list[-1],(1,2,0)).cpu()[:265]
    imsave(os.path.join(args.out_dir_p1_p2, "fake_img.png"), img)
     
    #visualize_training_results(img_list, G_losses, D_losses)

    
 