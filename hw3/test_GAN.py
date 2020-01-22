"""
This Module being called from .sh script for the problem 1
Runs the pre-trained GAN, DCGAN model and  saves the generated 32 fake images
"""
# general
import random
import os 
import numpy as np

# dl related
import torch

# custom modules
import parser
from generator import Generator
#from discriminator import Discriminator

# saving the subplot
import torchvision.utils as vutils
from scipy.misc import imsave 

if __name__=='__main__':
    num_gpu = 1 if torch.cuda.is_available() else 0

    args = parser.arg_parse()
    # set up the seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    #D = Discriminator(args.ngpu).eval()
    G = Generator(args.ngpu).eval()

    # load weights
    #D.load_state_dict(torch.load(args.resume))
    G.load_state_dict(torch.load(args.resume))
    if torch.cuda.is_available():
        #D = D.cuda()
        G = G.cuda()
        
    # encouraged to say 32 since you will generate 32 pics
    batch_size = args.batch_size
    latent_size = args.nz

    fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
    if torch.cuda.is_available():
        fixed_noise = fixed_noise.cuda()


    with torch.no_grad():
        fake =G(fixed_noise).detach().cpu()
    img = vutils.make_grid(fake, padding=2, normalize=True)

    # Save the fake image
    img = np.transpose(img,(1,2,0)).cpu()[:265]
    imsave(os.path.join(args.out_dir_p1_p2, 'fig1_2.jpg'), img)
 


