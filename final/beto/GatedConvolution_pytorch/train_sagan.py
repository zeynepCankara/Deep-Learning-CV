import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.gatedconv import InpaintGCNet, InpaintDirciminator
from models.sa_gan import InpaintSANet, InpaintSADirciminator
from models.loss import SNDisLoss, SNGenLoss, ReconLoss
from data.inpaint_dataset import InpaintDataset
from util.evaluation import AverageMeter
from evaluation import metrics
from PIL import Image
import pickle as pkl
import numpy as np
import logging
import time
import sys
import os
from test_images import validate
from util.util import load_consistent_state_dict

def train(netG, netD, GANLoss, ReconLoss, DLoss, optG, optD, dataloader, epoch, img_size, loss_writer):
    """
    Train Phase, for training and spectral normalization patch gan in
    Free-Form Image Inpainting with Gated Convolution (snpgan)

    """
    
    netG.train()
    netD.train()
    for i, (imgs, masks, _, _, _) in enumerate(dataloader):
        # masks = masks['val']

        # Optimize Discriminator
        optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()

        align_corners=True
        # imgs = F.interpolate(imgs, img_size, mode='bicubic', align_corners=align_corners)
        # imgs = imgs.clamp(min=-1, max=1)
        # masks = F.interpolate(masks, img_size, mode='bicubic', align_corners=align_corners)
        # masks = (masks > 0).type(torch.FloatTensor)
        
        imgs, masks = imgs.cuda(), masks.cuda()

        coarse_imgs, recon_imgs = netG(imgs, masks)
        complete_imgs = recon_imgs * masks + imgs * (1 - masks)

        pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

        pred_pos_neg = netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        d_loss = DLoss(pred_pos, pred_neg)
        # losses['d_loss'].update(d_loss.item(), imgs.size(0))
        d_loss_val = d_loss.item()
        d_loss.backward(retain_graph=True)

        optD.step()


        # Optimize Generator
        optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
        pred_neg = netD(neg_imgs)
        #pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
        g_loss = GANLoss(pred_neg)
        r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

        whole_loss = g_loss + r_loss

        # Update the recorder for losses
        # losses['g_loss'].update(g_loss.item(), imgs.size(0))
        # losses['r_loss'].update(r_loss.item(), imgs.size(0))
        # losses['whole_loss'].update(whole_loss.item(), imgs.size(0))
        g_loss_val = g_loss.item()
        r_loss_val = r_loss.item()
        whole_loss_val = whole_loss.item()
        whole_loss.backward()

        optG.step()
        if (i+1) % 25 == 0:
            print("Epoch {0} [{1}/{2}]:   Whole Loss:{whole_loss:.4f}   "
                        "Recon Loss:{r_loss:.4f}   GAN Loss:{g_loss:.4f}   D Loss:{d_loss:.4f}" \
                        .format(epoch, i+1, len(dataloader), whole_loss=whole_loss_val, r_loss=r_loss_val \
                        ,g_loss=g_loss_val, d_loss=d_loss_val))
        loss_writer.writerow([epoch,whole_loss_val, r_loss_val, g_loss_val, d_loss_val])
            


def main(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    dataset_type = args.dataset

    # Dataset setting
    train_dataset = InpaintDataset(args.train_image_list,\
                                      {'val':args.train_mask_list},
                                      mode='train', img_size=args.img_shape)
    train_loader = train_dataset.loader(batch_size=args.batch_size, shuffle=True,
                                            num_workers=4,pin_memory=True)

    val_dataset = InpaintDataset(args.val_image_list,\
                                      {'val':args.val_mask_list},
                                      # {'val':args.val_mask_list},
                                      mode='val', img_size=args.img_shape)
    val_loader = val_dataset.loader(batch_size=1, shuffle=False,
                                        num_workers=1)

    # Define the Network Structure
    netG = InpaintSANet()
    netD = InpaintSADirciminator()
    netG.cuda()
    netD.cuda()

    if args.load_weights != '':
        whole_model_path = args.load_weights
        nets = torch.load(whole_model_path)
        netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
        # netG.load_state_dict(netG_state_dict)
        load_consistent_state_dict(netG_state_dict, netG)
        netD.load_state_dict(netD_state_dict)

    # Define loss
    recon_loss = ReconLoss(*([1.2, 1.2, 1.2, 1.2]))
    gan_loss = SNGenLoss(0.005)
    dis_loss = SNDisLoss()
    lr, decay = args.learning_rate, 0.0
    optG = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=decay)
    optD = torch.optim.Adam(netD.parameters(), lr=4*lr, weight_decay=decay)

    best_score = 0

    # Create loss and acc file
    loss_writer = csv.writer(open(os.path.join(args.logdir, 'loss.csv'),'w'), delimiter=',')
    acc_writer = csv.writer(open(os.path.join(args.logdir, 'acc.csv'),'w'), delimiter=',')



    # Start Training
    for i in range(args.epochs):
        #train data
        train(netG, netD, gan_loss, recon_loss, dis_loss, optG, optD, train_loader, i+1, args.img_shape, loss_writer)

        # validate
        output_dir = os.path.join(args.result_dir,str(i+1))
        mse, ssim = validate(netG, val_loader, args.img_shape, output_dir, args.gt_dir)
        score = 1 - mse/100 + ssim
        print('MSE: ', mse, '     SSIM:', ssim, '     SCORE:', score)
        acc_writer.writerow([i+1,mse,ssim,score])

        
        saved_model = {
            'epoch': i + 1,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            # 'optG' : optG.state_dict(),
            # 'optD' : optD.state_dict()
        }
        torch.save(saved_model, '{}/epoch_{}_ckpt.pth.tar'.format(args.logdir, i+1))
        if score > best_score:
            torch.save(saved_model, '{}/best_ckpt.pth.tar'.format(args.logdir, i+1))
            best_score = score
            print('New best score at epoch', i+1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, default='logs', help='')
    parser.add_argument('--gt_dir', type=str, default='InpaintBenchmark/dlcv_gt_srgb', help='')
    parser.add_argument('--result_dir', type=str, default='results', help='')
    parser.add_argument('--dataset', type=str, default='places2', help='')
    parser.add_argument('--train_image_list', type=str, default='TrainImgs/gt_srgb.txt', help='')
    parser.add_argument('--train_mask_list', type=str, default='TrainImgs/masks.txt', help='')
    parser.add_argument('--val_image_list', type=str, default='InpaintBenchmark/dlcv_list.txt', help='')
    parser.add_argument('--val_mask_list', type=str, default='InpaintBenchmark/dlcv_mask.txt', help='')
    # parser.add_argument('--train_image_list', type=str, default='TrainImgs/gt_small.txt', help='')
    # parser.add_argument('--train_mask_list', type=str, default='TrainImgs/masks_small.txt', help='')
    # parser.add_argument('--val_image_list', type=str, default='InpaintBenchmark/dlcv_list_small.txt', help='')
    # parser.add_argument('--val_mask_list', type=str, default='InpaintBenchmark/dlcv_mask_small.txt', help='')
    parser.add_argument('--img_shape', type=int, default=256, help='')
    parser.add_argument('--model_path', type=str, default='model_logs/pretrained.pth.tar', help='')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    parser.add_argument('--batch_size', type=int, default=4, help='')
    parser.add_argument('--epochs', type=int, default=300, help='')
    # parser.add_argument('--load_weights', type=str, default='', help='')
    parser.add_argument('--load_weights', type=str, default='model_logs/pretrained.pth.tar', help='')
    
    # parser.add_argument('--', type=str, default='', help='')
    # parser.add_argument('--', type=int, default=, help='')
    # parser.add_argument('--', type=float, default=, help='')

    args = parser.parse_args()
    main(args)
