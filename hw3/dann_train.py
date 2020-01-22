# system related
import random
import os

# torch related
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data

# torch dataset related
from torchvision import datasets
from torchvision import transforms

import numpy as np

# model
from dann import DANN
from improved_dann import DANN_IMPROVED
# for the dataloader
from data_dann_train import DATA_DANN_TRAIN
# for arguments
import parser

if __name__=='__main__':
    """
    Configuration parameters for arguments
    args.source: mnistm
    args.target: svhn
    args.data_dir: './hw3_data_new/digits/'
    args.image_size: 28
    args.load: mnsitm
    args.mode: train
    args.lr: 0.0001
    args.batch_size: 128
    args.num_epochs: 100

    Reference: https://github.com/fungtion/DANN-py3
    """
    # parse the arguments
    args = parser.arg_parse()

    # directory to save the model
    model_root = 'models'

    # Cuda available
    cuda = True
    cudnn.benchmark = True

    lr = args.lr #0.0001
    batch_size = args.batch_size #128
    image_size = args.image_size #28
    num_epochs = args.num_epochs #100
    workers = args.workers #4
    model_type = args.model_type #dann

    target = args.target
    source = args.source
    manual_seed = args.random_seed #999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
 

    args.load = args.source
    dataloader_source = torch.utils.data.DataLoader(DATA_DANN_TRAIN(args),
                                batch_size=batch_size, 
                                num_workers=workers,
                                shuffle=True)

    args.load = args.target
    dataloader_target = torch.utils.data.DataLoader(DATA_DANN_TRAIN(args),
                                batch_size=batch_size, 
                                num_workers=workers,
                                shuffle=True)

    # Load the DANN model
    if model_type == 'dann':
        dann = DANN()
    else:
        dann = DANN_IMPROVED()

    # Configure the optimizer

    adam = optim.Adam(dann.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        dann = dann.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for params in dann.parameters():
        params.requires_grad = True

    # training

    for epoch in range(num_epochs):

        dataset_length = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        idx = 0
        while idx < dataset_length:
            # Set up alpha
            a = float(idx + epoch * dataset_length) 
            a = a / num_epochs
            a = a / dataset_length

            alpha = 2. / (1. + np.exp(-10 * a)) - 1

            # Set up source data loader
            data_source = data_source_iter.next()
            s_img, s_label, s_path = data_source

            dann.zero_grad()
            batch_size = len(s_label)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_type = torch.LongTensor(batch_size)
            domain_type = torch.zeros(batch_size)
            domain_type = domain_type.long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                class_type = class_type.cuda()
                domain_type = domain_type.cuda()

            input_img.resize_as_(s_img).copy_(s_img)
            class_type.resize_as_(s_label).copy_(s_label)

            class_output, domain_output = dann(input_data=input_img, alpha=alpha)

            err_s_label = loss_class(class_output, class_type)
            err_s_domain = loss_domain(domain_output, domain_type)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _, t_path = data_target

            batch_size = len(t_img)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            domain_type = torch.ones(batch_size)
            domain_type = domain_type.long()

            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                domain_type = domain_type.cuda()

            input_img.resize_as_(t_img).copy_(t_img)

            _, domain_output = dann(input_data=input_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_type)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            adam.step()

            idx += 1
            print("==== EPOCH %d ==="%(epoch))
            print("[Iter: %d / Total: %d]"%(idx,dataset_length))
            print("[Error Source Label: %f]"%(err_s_label.data.cpu().numpy()))
            print("[Error Source Domain Label: %f]"%(err_s_domain.data.cpu().numpy()))
            print("[Error Target Domain Label: %f]"%(err_t_domain.data.cpu().item()))
        # save the model
        torch.save(dann, '{0}/dann_target_{1}_epoch_{2}.pth'.format(model_root, target, epoch))

    print('Training Finished...')