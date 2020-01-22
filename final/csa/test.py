import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from util.params import Opion
from util.generate_images import generate_images
from util.evaluate import get_average_mse_ssim
import torch.nn.functional as F
import warnings
import csv
from util.data_load import Data_load
from models.models import create_model

warnings.filterwarnings(action='once')
opt = Opion()

transform_mask = transforms.Compose([
    # transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
    ])
transform = transforms.Compose([
    #  transforms.Resize((opt.fineSize,opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

dataset_test = Data_load(opt.valroot, opt.valmask, transform, transform_mask)
iterator_test = (data.DataLoader(dataset_test, batch_size=opt.batchSize,shuffle=False))
print(len(dataset_test))
model = create_model(opt)
total_steps = 0

load_epoch=24
model.load(load_epoch)

f = open('evaluations.csv','w+')
writer = csv.writer(f)
# writer.writerow(['epoch','mse_avg','ssim_avg'])

generate_images(model, iterator_test)
gts = []
preds = []
for i in range(100):
    img_name = "{}.jpg".format(401+i)
    gts.append(os.path.join(opt.valroot,img_name))
    pred_name = "{}.jpg".format(401+i)
    preds.append(os.path.join("pred",pred_name))

mse_avg, ssim_avg = get_average_mse_ssim(gts,preds)
score = 1 - mse_avg/100 + ssim_avg

print(f"MSE:{mse_avg} SSIM:{ssim_avg}, score:{score}")
writer.writerow(['{load_epoch},{mse_avg},{ssim_avg}'])

f.close()