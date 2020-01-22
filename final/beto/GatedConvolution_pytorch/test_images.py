import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sa_gan import InpaintSANet, InpaintSADirciminator
from models.loss import SNDisLoss, SNGenLoss, ReconLoss, PerceptualLoss, StyleLoss
from data.inpaint_dataset import InpaintDataset
from util.util import load_consistent_state_dict

from PIL import Image
import numpy as np
import os
import argparse
# from evaluate import EvaluateImages

def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy().round()


def ProcessImg(crop_size, x_pos, y_pos, ori_imgs, ori_masks, img_size, netG):

    input_img = ori_imgs[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)]
    input_mask = ori_masks[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)]
    input_img = torch.unsqueeze(input_img,0)
    input_mask = torch.unsqueeze(input_mask,0)

    align_corners=True
    # Resize the images
    # ori_imgs = ori_imgs - 0.1
    # ori_imgs = ori_imgs.clamp(min=-1, max=1)
    imgs = F.interpolate(input_img, img_size, mode='bicubic', align_corners=align_corners)
    masks = F.interpolate(input_mask, img_size, mode='bicubic', align_corners=align_corners)

    # Limit the min & max value of imgs cause the interpolation can produce higher results
    imgs = imgs.clamp(min=-1, max=1)
    masks = (masks > 0).type(torch.FloatTensor)
    
    imgs, masks = imgs.cuda(), masks.cuda()

    recon_imgs, x = netG(imgs, masks)
    # recon_imgs = x

    complete_imgs = recon_imgs
    # complete_imgs = recon_imgs * masks + imgs * (1 - masks)

    # gen_img = img2photo(complete_imgs)

    # Make the images of the original input size.
    # imgs = F.interpolate(imgs, (ori_imgs.shape[2], ori_imgs.shape[3]), mode='bicubic', align_corners=align_corners)
    # imgs = imgs.clamp(min=-1, max=1)
    complete_imgs = F.interpolate(complete_imgs, (crop_size, crop_size), mode='bicubic', align_corners=align_corners)
    # complete_imgs = F.interpolate(complete_imgs, (ori_imgs.shape[2], ori_imgs.shape[3]), mode='bicubic', align_corners=align_corners)
    complete_imgs = complete_imgs.clamp(min=-1, max=1)




    # color correction
    complete_imgs_clone = complete_imgs.clone()
    val = 0.895
    complete_imgs[0,0,:,:] = complete_imgs[0,0,:,:] + 0.03
    complete_imgs[0,1,:,:] = complete_imgs[0,1,:,:] + 0.01
    complete_imgs[0,2,:,:] = complete_imgs[0,2,:,:] - 0.02
    complete_imgs = complete_imgs * input_mask * (val+0.02) + complete_imgs_clone * (1 - input_mask) * val + complete_imgs*(1-val)
    # complete_imgs = complete_imgs * original_mask + complete_imgs_clone * (1 - original_mask)# + complete_imgs*(1-val)
    complete_imgs = complete_imgs.clamp(min=-1, max=1)




    # # complete_imgs[complete_imgs>-2] = 1
    # val = 0.895
    # complete_imgs[0,0,:,:] = complete_imgs[0,0,:,:] + 0.03
    # complete_imgs[0,1,:,:] = complete_imgs[0,1,:,:] + 0.01
    # complete_imgs[0,2,:,:] = complete_imgs[0,2,:,:] - 0.02
    # complete_imgs = complete_imgs * original_mask * (val+0.02) + ori_imgs * (1 - original_mask) * val + complete_imgs*(1-val)
    # complete_imgs = complete_imgs.clamp(min=-1, max=1)


    # output_img[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)] = complete_imgs
    # print('aaaa', complete_imgs.shape)
    return complete_imgs




def validate(netG, dataloader, img_size, result_dir, gt_dir):
    netG.cuda()
    netG.eval()

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with torch.no_grad():
        for i, (ori_imgs, dilated_mask, original_mask, image_name, img_path) in enumerate(dataloader):
            cnn_mask = dilated_mask
            output_img = ori_imgs.clone()
            # print(ori_imgs.shape)
            crop_size = 256

            ori_imgs, cnn_mask, output_img, original_mask = ori_imgs.cuda(), cnn_mask.cuda(), output_img.cuda(), original_mask.cuda()



            # Reconstruct the image in batches of 256x256

            for j in range (ori_imgs.shape[2]//crop_size):
                x_pos = crop_size * j
                for k in range (ori_imgs.shape[3]//crop_size):
                    y_pos = crop_size * k
                    # print(x_pos, y_pos)
                    complete_imgs = ProcessImg(crop_size, x_pos, y_pos, ori_imgs, cnn_mask, img_size, netG)
                    output_img[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)] = complete_imgs
                    
                y_pos = ori_imgs.shape[3] - crop_size
                # print(x_pos, y_pos)
                complete_imgs = ProcessImg(crop_size, x_pos, y_pos, ori_imgs, cnn_mask, img_size, netG)
                output_img[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)] = complete_imgs
                
            x_pos = ori_imgs.shape[2] - crop_size
            for k in range (ori_imgs.shape[3]//crop_size):
                y_pos = crop_size * k
                # print(x_pos, y_pos)
                complete_imgs = ProcessImg(crop_size, x_pos, y_pos, ori_imgs, cnn_mask, img_size, netG)
                output_img[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)] = complete_imgs
                
            y_pos = ori_imgs.shape[3] - crop_size
            # print(x_pos, y_pos)
            complete_imgs = ProcessImg(crop_size, x_pos, y_pos, ori_imgs, cnn_mask, img_size, netG)
            output_img[0,:,x_pos:(x_pos+crop_size),y_pos:(y_pos+crop_size)] = complete_imgs
                
            # print(output_img.shape)

            # output_img[:,output_img[:
            # print(output_img.shape)
            white_removed = torch.sum(output_img, 1)
            white_removed = (white_removed[0] > 2.6).type(torch.FloatTensor).cuda()
            output_img[0,0,:,:] = ori_imgs[0,0,:,:] * white_removed + output_img[0,0,:,:] * (1-white_removed)
            output_img[0,1,:,:] = ori_imgs[0,1,:,:] * white_removed + output_img[0,1,:,:] * (1-white_removed)
            output_img[0,2,:,:] = ori_imgs[0,2,:,:] * white_removed + output_img[0,2,:,:] * (1-white_removed)


            # pink_removed = (ori_imgs[0,0,:,:] > 0.9) * (ori_imgs[0,1,:,:] < -0.9) * (ori_imgs[0,2,:,:] > 0.9)
            # pink_removed = pink_removed.type(torch.FloatTensor).cuda()
            # output_img[0,0,:,:] = ori_imgs[0,0,:,:] * pink_removed + output_img[0,0,:,:] * (1-pink_removed)
            # output_img[0,1,:,:] = ori_imgs[0,1,:,:] * pink_removed + output_img[0,1,:,:] * (1-pink_removed)
            # output_img[0,2,:,:] = ori_imgs[0,2,:,:] * pink_removed + output_img[0,2,:,:] * (1-pink_removed)
            # print(white_removed.shape)
            # d=d













            align_corners=True
            # Resize the images
            # ori_imgs = ori_imgs - 0.1
            # ori_imgs = ori_imgs.clamp(min=-1, max=1)
            imgs = F.interpolate(ori_imgs, img_size, mode='bicubic', align_corners=align_corners)
            masks = F.interpolate(original_mask, img_size, mode='bicubic', align_corners=align_corners)

            # Limit the min & max value of imgs cause the interpolation can produce higher results
            imgs = imgs.clamp(min=-1, max=1)
            masks = (masks > 0).type(torch.FloatTensor)
            
            imgs, masks = imgs.cuda(), masks.cuda()

            recon_imgs, x = netG(imgs, masks)
            # recon_imgs = x

            complete_imgs = recon_imgs * masks + imgs * (1 - masks)


            # Make the images of the original input size.
            complete_imgs = F.interpolate(complete_imgs, (ori_imgs.shape[2], ori_imgs.shape[3]), mode='bicubic', align_corners=align_corners)
            complete_imgs = complete_imgs.clamp(min=-1, max=1)
            complete_imgs_clone = complete_imgs.clone()

            # color correction
            val = 0.895
            complete_imgs[0,0,:,:] = complete_imgs[0,0,:,:] + 0.03
            complete_imgs[0,1,:,:] = complete_imgs[0,1,:,:] + 0.01
            complete_imgs[0,2,:,:] = complete_imgs[0,2,:,:] - 0.02
            complete_imgs = complete_imgs * original_mask * (val+0.02) + complete_imgs_clone * (1 - original_mask) * val + complete_imgs*(1-val)
            # complete_imgs = complete_imgs * original_mask + complete_imgs_clone * (1 - original_mask)# + complete_imgs*(1-val)
            complete_imgs = complete_imgs.clamp(min=-1, max=1)

















            # gen_image = output_img


            weight = 0.5
            gen_image = ori_imgs*weight*(1 - original_mask) + complete_imgs*original_mask*weight + output_img * (1 - weight) 
            gen_image = gen_image.clamp(min=-1, max=1)





            comp_img = img2photo(gen_image)[0]
            comp_img = Image.fromarray(comp_img.astype(np.uint8))

            # # Save images
            # real_img = img2photo(imgs)
            # ori_imgs = img2photo(ori_imgs)
            # real_img = Image.fromarray(real_img[0].astype(np.uint8))
            # gen_img = Image.fromarray(gen_img[0].astype(np.uint8))
            # ori_imgs = Image.fromarray(ori_imgs[0].astype(np.uint8))

            
            ####################################################################
            ####################################################################
            ####################################################################
            ####################################################################
            ##### CHANGE TO THE ORIGINAL IMAGE NAME THIS BEFORE SUBMITTING #####
            ####################################################################
            ####################################################################
            ####################################################################
            ####################################################################
            # image_name = image_name[0].split('.')[0]
            image_name = image_name[0].split('_')[0]

            input_base = Image.open('./color.jpg')
            # input_img = Image.open(img_path[0])

            # gen_img.save(os.path.join(result_dir, "{}_gen.jpg".format(image_name))) # Output before post-process
            comp_img.save(os.path.join(result_dir, "{}.jpg".format(image_name)),
                format = 'JPEG', quality = 100, icc_profile = input_base.info.get('icc_profile',''))
            # real_img.save(os.path.join(result_dir, "{}_real.jpg".format(image_name)))
            # ori_imgs.save(os.path.join(result_dir, "{}.jpg".format(image_name)))
            #     format = 'JPEG', quality = 100, icc_profile = input_base.info.get('icc_profile',''))
            
    # mse, ssim = EvaluateImages(gt_dir, result_dir)
    
    # return mse, ssim
def main(args):

    # Dataset setting
    val_dataset = InpaintDataset(args.source_dir,
                                    # {'val':args.mask_list},
                                    mode='val')
    val_loader = val_dataset.loader(batch_size=1, shuffle=False,
                                        num_workers=1)

    ### Generate a new val data

    # Define the Network Structure
    whole_model_path = args.model_path
    nets = torch.load(whole_model_path)
    # netG_state_dict = nets['netG_state_dict']
    netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
    netG = InpaintSANet()
    # load_consistent_state_dict(netG_state_dict, netG)
    netG.load_state_dict(netG_state_dict)


    
    # mse, ssim = 
    validate(netG, val_loader, args.img_shape, args.result_dir, args.gt_dir)
    # print('MSE: ',mse, '     SSIM:', ssim, '     SCORE:', 1 - mse/100 + ssim)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_dir', type=str, default='InpaintBenchmark/dlcv_gt', help='')
    # parser.add_argument('--gt_dir', type=str, default='InpaintBenchmark/dlcv_gt_srgb', help='')
    parser.add_argument('--source_dir', type=str, default='./aaaaa', help='')
    parser.add_argument('--result_dir', type=str, default='results', help='')
    # parser.add_argument('--dataset', type=str, default='places2', help='')
    # parser.add_argument('--image_list', type=str, default='InpaintBenchmark/dlcv_list.txt', help='')
    # parser.add_argument('--image_list', type=str, default='InpaintBenchmark/dlcv_masked_from_gt.txt', help='')
    # parser.add_argument('--image_list', type=str, default='InpaintBenchmark/dlcv_masked_srgb.txt', help='')
    # parser.add_argument('--image_list', type=str, default='InpaintBenchmark/dlcv_gt.txt', help='')
    # parser.add_argument('--image_list', type=str, default='InpaintBenchmark/dlcv_list_small.txt', help='')
    # parser.add_argument('--mask_list', type=str, default='InpaintBenchmark/dlcv_mask.txt', help='')

    # parser.add_argument('--train_image_list', type=str, default='TrainImgs/gt_small.txt', help='')
    # parser.add_argument('--train_mask_list', type=str, default='TrainImgs/masks_small.txt', help='')
    parser.add_argument('--img_shape', type=int, default=256, help='')
    # parser.add_argument('--model_path', type=str, default='logs/epoch_44_ckpt.pth.tar', help='')
    parser.add_argument('--model_path', type=str, default='./best_ckpt.pth.tar', help='')
    # parser.add_argument('--model_path', type=str, default='logs_old/8_train_cropped_256x256/best_ckpt.pth.tar', help='')
    # parser.add_argument('--model_path', type=str, default='model_logs/pretrained.pth.tar', help='')
    
    # parser.add_argument('--', type=str, default='', help='')
    # parser.add_argument('--', type=int, default=, help='')
    # parser.add_argument('--', type=float, default=, help='')

    args = parser.parse_args()
    main(args)
