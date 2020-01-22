"""
    This module saves the predictions done by the pre-trained DANN model to the specified directory
"""
# core
import csv
import random
import os

# torch related
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from torchvision import datasets

# personal
import dann
from data_dann_train import DATA_DANN_TEST
from dann import DANN
from improved_dann import DANN_IMPROVED
import parser

 

def write_to_csv(pred_path, preds):
  """Writes the predictions to a csv file
  """
  with open(pred_path, mode='w') as csv_file:
      columns = ['image_name', 'label']
      writer = csv.DictWriter(csv_file, fieldnames = columns)

      writer.writeheader()
      for pred in preds: 
        file_name, label = pred
        writer.writerow({'image_name': file_name, 'label': label})


def extract_filename(filename):
  dot_pos = filename.find('.')
  file_name = filename[dot_pos-9:]
  return file_name

def test(dataset_name, model_dir, args):
    assert dataset_name in ['mnistm', 'svhn']

    cuda = True
    cudnn.benchmark = True
    batch_size = args.batch_size 
    image_size = 28
    alpha = 0
    model_type = args.model_type

    # cache the results in here!!!
    preds = []
    """load data"""

    dataloader = torch.utils.data.DataLoader(DATA_DANN_TEST(args),
                        batch_size=batch_size, 
                        num_workers=args.workers,
                        shuffle=True)


    """ training """
    # load the model according to the argument
    if model_type == 'dann':
        dann = DANN()
    else:
        dann = DANN_IMPROVED()
    dann = torch.load(model_dir)
    dann = dann.eval()

    if cuda:
        dann = dann.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_path = data_target

        batch_size = len(t_path)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            #t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()


        input_img.resize_as_(t_img).copy_(t_img)
        #class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = dann(input_data=input_img, alpha=alpha)
  
        # pred and the image path are both tensor having length same as the batch size
        for idx, prediction in enumerate(class_output):
          prediction = torch.argmax(prediction.squeeze(), dim=0).detach().cpu().numpy()
          file_name = extract_filename(t_path[idx])
          preds.append((file_name, int(prediction)))


        i += 1
    return preds



if __name__=='__main__':
    # retrieve arguments
    args = parser.arg_parse()

    manual_seed = args.random_seed #999
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)


    torch.cuda.set_device(args.gpu)

    # target dataset
    target = args.target
    # output predictions
    pred_path = args.pred_path
    # testing datset
    dataset_path = args.data_dir
    # model load model
    if target == 'mnistm':
        # target mnistm -> load the model which trained with source svhn
        model_path = args.resume_mnistm
    else: 
        # target svhn -> load the model which trained with source mnistm
        model_path = args.resume_svhn
    
    # obtain predictions 
    predictions = test(target, model_path, args)
    # write predictions to the csv
    write_to_csv(pred_path, predictions)
    




