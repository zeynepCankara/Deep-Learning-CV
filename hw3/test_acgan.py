"""
This Module being called from .sh script for the problem 2
Runs the pre-trained ACGAN model and  saves generates 10 pairs of images entangled via 'smiling feature attribute'
"""
# general
import random
import os 
import numpy as np

# dl related
import torch
from torch.autograd import Variable

# custom modules
import parser
from acgan import Generator_ACGAN

# saving the subplot
import torchvision.utils as vutils
 

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
    num_gpu = 1 if torch.cuda.is_available() else 0

    args = parser.arg_parse()
    # set up the seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # set up the model
    generator = Generator_ACGAN().cuda()

    # load weights
    generator.load_state_dict(torch.load(args.resume))
    generator.eval()
        
    # encouraged to say 32 since you will generate 32 pics
    batch_size = args.batch_size
    latent_size = args.nz

    fixed_input = random_generator_input(num_pairs = 10)
    # generate the random input
    img = generator(fixed_input)
    generator.train()
   
    vutils.save_image(img.cpu().data, os.path.join(args.out_dir_p1_p2, 'fig2_2.jpg') ,nrow = 10)