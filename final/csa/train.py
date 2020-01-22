import time
import torch.nn.functional as F
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from util.params import Opion
from util.generate_images import generate_images
from util.evaluate import get_average_mse_ssim
import warnings

opt = Opion()
warnings.filterwarnings(action='once')

def train():
    '''load data'''
    transform_mask = transforms.Compose([
    # transforms.Resize((opt.fineSize,opt.fineSize)),
    transforms.ToTensor(),
    ])
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.Resize((opt.fineSize,opt.fineSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    dataset_train = Data_load(opt.dataroot, opt.maskroot, transform, transform_mask)
    iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize,shuffle=True))

    dataset_test = Data_load(opt.valinputroot, opt.valmask, transform, transform_mask)
    iterator_test = (data.DataLoader(dataset_test, batch_size=opt.batchSize,shuffle=False))

    print(len(dataset_train))
    print(len(dataset_test))
    model = create_model(opt)
    total_steps = 0

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    '''if resuming'''
    # model.load(epoch_number/best)

    best_score = 0
    gts = []
    preds = []
    for i in range(100):
        img_name = "{}.jpg".format(401+i)
        gts.append(os.path.join(opt.valroot,img_name))
        pred_name = "{}.jpg".format(401+i)
        preds.append(os.path.join("pred",pred_name))

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # for epoch in range(loaded_epoch_#+1, opt.niter + opt.niter_decay + 1):
        print("this is a new epoch!")
        epoch_start_time = time.time()
        epoch_iter = 0

        for ori_image, ori_mask in iterator_train:
            align_corners=True
            img_size=512 # Or any other size

            # Shrinking the image to a square. # ori_imgs is the original image
            image = F.interpolate(ori_image, img_size, mode='bicubic', align_corners=align_corners)
            mask = F.interpolate(ori_mask, img_size, mode='bicubic', align_corners=align_corners) # mask looks pretty complete at this stage
            
            image = image.clamp(min=-1, max=1)
            mask  = (mask>0).type(torch.FloatTensor)

            image=image.cuda()
            mask=mask.cuda()
            mask=mask[0][0]
            mask=torch.unsqueeze(mask,0)
            mask=torch.unsqueeze(mask,1)
            mask=mask.byte()

            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(image,mask) # sets both input data with mask and latent mask.
            model.set_gt_latent()
            model.optimize_parameters()

            if total_steps % opt.display_freq== 0:
                real_A,real_B,fake_B=model.get_current_visuals()
                # real_A=input, real_B=ground truth fake_b=output
                pic = (torch.cat([real_A, real_B,fake_B], dim=0) + 1) / 2.0
                print("saving image!")
                torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (
                opt.checkpoints_dir, epoch, total_steps + 1, len(dataset_train)), nrow=2)

                if total_steps %1== 0:
                    errors = model.get_current_errors()
                    print(errors)


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save(epoch)
        
        # evaluation #
        generate_images(model,iterator_test)
        mse_avg, ssim_avg = get_average_mse_ssim(gts,preds)
        score = 1 - mse_avg/100 + ssim_avg

        if score > best_score:
            best_score = score
            model.save("best")
            print(f"best_score {best_score} found in epoch {epoch}. MSE:{mse_avg} SSIM:{ssim_avg}")
            
        ### end evaluation ###
        
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()

if __name__ == '__main__':
    train()