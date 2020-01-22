import os
import parser
import models

import data
import data_test 

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score
import skimage

from mean_iou_evaluate import mean_iou_score
# import torch library
import torch

# models
import simple_baseline_model
import baseline_model

def prediction_labeller(idx):
    remain = '0' * (4-len(idx))
    idx = remain + idx
    return str(idx)

# change the hard coded path $2 -> output dir
# output_dir = "./preds/"

def evaluate(model, data_loader, mode = "train", output_dir = "./preds/"):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        cnt = 0
        for _, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            if mode == "val" or mode == "test":
                # save images only during test and validation
                for p in pred:
                    p = torch.argmax(p.squeeze(), dim=0).detach().cpu().numpy()
                    skimage.io.imsave(os.path.join(output_dir, prediction_labeller(str(cnt)) + ".png"), p)
                    cnt += 1
                pass
            else:
                # no need to save during training
                pass

            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return mean_iou_score(gts, preds)

def evaluate_test(model, data_loader, output_dir = "./preds/"):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
 
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        cnt = 0
        for _, imgs in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            # save images only during test and validation
            for p in pred:
                p = torch.argmax(p.squeeze(), dim=0).detach().cpu().numpy()
                skimage.io.imsave(os.path.join(output_dir, prediction_labeller(str(cnt)) + ".png"), p)
                cnt += 1
            pass

            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            
            preds.append(pred)
         
    preds = np.concatenate(preds)
    return 0



if __name__ == '__main__':

    args = parser.arg_parse()

    # get input and output directory
    input_dir = args.input_dir

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    if input_dir == "val_test":
        test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                                  batch_size=args.test_batch,
                                                  num_workers=args.workers,
                                                  shuffle=False)
    else:
        test_loader = torch.utils.data.DataLoader(data_test.DATA_TEST(args, mode='test'),
                                                  batch_size=args.test_batch,
                                                  num_workers=args.workers,
                                                  shuffle=False)
    ''' prepare mode '''
    if args.model == "simple_baseline":
        model = simple_baseline_model.SimpleBaselineModel(args).cuda()
    else:
        model = baseline_model.BaselineModel(args).cuda()

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    if input_dir == "val_test":
        acc = evaluate(model, test_loader, mode="val", output_dir = args.output_dir)
        print('Testing Accuracy: {}'.format(acc))
    else:
        _ = evaluate_test(model, test_loader, output_dir = args.output_dir)
