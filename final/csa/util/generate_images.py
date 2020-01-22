import torch.nn.functional as F
import torch
import torchvision
from util.unnormalise import UnNormalize
import numpy as np
from PIL import Image
from util.params import Opion
import os


unnorm = UnNormalize(mean=[0.5] * 3, std=[0.5] * 3)
opt = Opion()

def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy().round()

input_image = Image.open(os.path.join(opt.dataroot,'001.jpg'))
def save_jpg(image, image_name, target_dir='pred'):
    image = img2photo(image)
    image = Image.fromarray(image[0].astype(np.uint8))
    image.save(os.path.join(target_dir, "{}.jpg".format(image_name)),
    format = 'JPEG', quality = 100, icc_profile = input_image.info.get('icc_profile',''))

def generate_images(model, dataloader, pred_dir='pred/'):
  # model.eval()
  # print("generating images")
  for idx, (ori_image, ori_mask) in enumerate(dataloader):
      align_corners=True
      img_size=512 # Or any other size

      # Shrinking the image to a square. # ori_imgs is the original image
      image = F.interpolate(ori_image, img_size, mode='bicubic', align_corners=align_corners)
      mask = F.interpolate(ori_mask, img_size, mode='bicubic', align_corners=align_corners)
      image = image.clamp(min=-1, max=1)
      
      image=image.cuda()
      mask=mask.cuda()
      mask=mask[0][0]
      mask=torch.unsqueeze(mask,0)
      mask=torch.unsqueeze(mask,1)
      mask=mask.byte()
      
      model.set_input(image,mask)
      model.set_gt_latent()
      model.test()

      ori_image=ori_image.cuda()
      ori_mask=ori_mask.cuda()
      ori_height, ori_width = (ori_image.shape[2], ori_image.shape[3])

      fake_B=model.get_current_visuals(fake=True)
      fake_img=fake_B*mask + image*(1-mask)
      
      # make image big again
      new_img = F.interpolate(fake_img, (ori_height, ori_width), mode='bicubic', align_corners=align_corners)
      new_img = new_img.clamp(min=-1,max=1)
      # print(ori_image.shape, ori_mask.shape, new_img.shape)

      new_img = ori_image*(1-ori_mask)+new_img*ori_mask  #when new_img*ori_mask, information is lost.
      new_img =  unnorm(new_img)

      save_jpg(new_img,image_name=401+idx)
