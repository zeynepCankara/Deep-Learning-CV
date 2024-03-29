import os
import torch

import parser
import models
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from test import evaluate

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)


if __name__=='__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)
    val_loader   = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    # convert PIL image to tensor
    print(type(train_loader))

    ''' load model '''
    print('===> prepare model ...')

    # NOTE: this is a testing code
    load_prev_model = args.load_prev_model

    if args.model == 'simple_baseline':

        if load_prev_model:
            model = models.SimpleBaselineModel(args).cuda()
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            model = models.SimpleBaselineModel(args)
            model.cuda() # load model to gpu

    if args.model == 'baseline':

        if load_prev_model:
            model = models.BaselineModel(args).cuda()
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            model = models.BaselineModel(args)
            model.cuda() # load model to gpu


    if args.model == 'best':
        if load_prev_model:
                model = models.BestModel(args).cuda()
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint)
        else:
                model = models.BestModel(args)
                model.cuda() # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0

    # MODIF AFTER MY TRAININ FINISHES
    best_acc = args.prev_best_acc

    for epoch in range(1, args.epoch+1):
        # simple baseline performance surpassed
        model.train()

        for idx, (imgs, cls) in enumerate(train_loader):

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, cls = imgs.cuda(), cls.cuda()

            ''' forward path '''
            output = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
     
         
            #cls = torch.reshape(cls, (args.train_batch, 352, 448)).long()

            loss = criterion(output, cls) # compute loss


            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            # change gpu
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)

            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        # DO NOT save the model every iteration
        #save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
