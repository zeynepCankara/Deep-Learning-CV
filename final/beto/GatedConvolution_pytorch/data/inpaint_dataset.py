import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
# from .base_dataset import BaseDataset#, NoriBaseDataset
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
from scipy import ndimage
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random

import PIL.ImageCms

ALLMASKTYPES = ['bbox', 'seg', 'random_bbox', 'random_free_form', 'val']

def transform_train(image, type):
    if random.random() > 0.5:
        image = TF.hflip(image) 
    angle = random.choice([0,90,180,270])
    image = TF.rotate(image, angle, expand=True)

    # Random crop 3/4 of the images
    # if type == 'img':
    if type == 'img' and random.random() <= 0.75:
        w,h = image.size
        width = random.randint(256,w)
        height = random.randint(256,h)
        # width = 256
        # height = 256
        left = random.randint(0,w-width)
        top = random.randint(0,h-height)
        image = TF.crop(image, top, left, height, width)
    image = TF.to_tensor(image)
    return image

class InpaintDataset(Dataset):
    """
    Dataset for Inpainting task
    Params:
        img_flist_path(str): The file which contains img file path list (e.g. test.flist)
        mask_flist_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
        random_bbox_shape(tuple): if use random bbox mask, it define the shape of the mask (default:(32,32))
        random_bbox_margin(tuple): if use random bbox, it define the margin of the bbox which means the distance between the mask and the margin of the image
                                    (default:(64,64))
    Return:
        img, *mask
    """
    def __init__(self, img_flist_path, mode='val', img_size=256,
                resize_shape=(256, 256), 
                transforms_oprs=['to_tensor'], #transforms_oprs=['random_crop', 'to_tensor'],
                # random_bbox_shape=(32, 32), random_bbox_margin=(64, 64),
                random_bbox_shape=(64, 64), random_bbox_margin=(1, 1),
                random_ff_setting={'img_shape':[256,256],'mv':5, 'ma':4.0, 'ml':40, 'mbw':10}, random_bbox_number=6):


        test_dir = sorted(os.listdir(img_flist_path))
        self.img_paths = []
        self.mask_paths = []
        for file_name in test_dir:
            if file_name.endswith('masked.jpg'):
                self.img_paths.append(os.path.join(img_flist_path, file_name))
            else:
                self.mask_paths.append(os.path.join(img_flist_path, file_name))
      



        # with open(img_flist_path, 'r') as f:
        #     self.img_paths = f.read().splitlines()

        # self.mask_paths = {}
        # for mask_type in mask_flist_paths_dict:
        #     assert mask_type in ALLMASKTYPES
        #     if 'random' in mask_type:
        #         self.mask_paths[mask_type] = ['' for i in self.img_paths]
        #     else:
        #         with open(mask_flist_paths_dict[mask_type]) as f:
        #             self.mask_paths[mask_type] = f.read().splitlines()

        self.mode = mode
        self.img_size = img_size
        self.resize_shape = resize_shape
        self.random_bbox_shape = random_bbox_shape
        self.random_bbox_margin = random_bbox_margin
        self.random_ff_setting = random_ff_setting
        self.random_bbox_number = random_bbox_number
        
        self.transform_val = transforms.Compose([
                        # transforms.ColorJitter(brightness=0, contrast=0, saturation=0.1, hue=0),
                        transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                        ])


    def loader(self, **args):
        return DataLoader(dataset=self, **args)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # create the paths for images and masks

        img_path = self.img_paths[index]
        if self.mode=='val':
            mask_index = index
        else: # Pick a random mask during training
            mask_index = random.randint(0, len(self.img_paths)-1)

        image_name = os.path.basename(img_path)

        mask_paths = self.mask_paths[mask_index]
        # mask_paths = {}
        # for mask_type in self.mask_paths:
        #     mask_paths[mask_type] = self.mask_paths[mask_index]

        if self.mode=='val':

            img = Image.open(img_path)

            img = self.transform_val(img)
            # input_img = Image.open('color.jpg')
            # img.save('recolor.jpg',
            #     format = 'JPEG', quality = 100, icc_profile = input_img.info.get('icc_profile',''))
            # img = self.transform_val(Image.open('recolor.jpg'))

            img = (img/0.5 - 1) # Make image in range [-1,1] 

            # mask, original_mask = self.read_mask(mask_paths['val'], 'val')
            mask, original_mask = self.read_mask(mask_paths, 'val')
            mask = self.transform_val(mask).type(torch.FloatTensor)[:1, :,:] 
            original_mask = self.transform_val(original_mask).type(torch.FloatTensor)[:1, :,:] 
        else:
            img = transform_train(Image.open(img_path), 'img')
            mask_type = random.choice(['val','random_bbox','random_free_form','val'])
            # mask = self.read_mask('TrainImgs/train_masks/001.jpg', 'val')
            # mask = self.read_mask(mask_paths['val'], 'random_free_form')
            # mask.save('expanded_masks/'+image_name)
            # mask = transform_train(self.read_mask('TrainImgs/train_masks/001.jpg', 'random_bbox'),'mask')
            # mask = {mask_type:(transform_train(self.read_mask(mask_paths[mask_type], mask_type), 'mask')).type(torch.FloatTensor)[:1, :,:] for mask_type in mask_paths}
            if(mask_type=='val'):
                # mask, original_mask = self.read_mask(mask_paths[mask_type], mask_type)
                mask, _ = self.read_mask(mask_paths[mask_type], mask_type)
                mask = (transform_train(mask, 'mask')).type(torch.FloatTensor)[:1, :,:] 
                # original_mask = (transform_train(original_mask, 'mask')).type(torch.FloatTensor)[:1, :,:] 
            else:
                mask = (transform_train(self.read_mask('', mask_type), 'mask')).type(torch.FloatTensor)[:1, :,:] 
                # original_mask = mask.copy()

            # Rescale the images
            img = torch.unsqueeze(img,0)
            mask = torch.unsqueeze(mask,0)
            # original_mask = torch.unsqueeze(original_mask,0)
            align_corners = True
            img = (img/0.5 - 1) # Make image in range [-1,1]     
            img = F.interpolate(img, self.img_size, mode='bicubic', align_corners=align_corners)
            img = img.clamp(min=-1, max=1)
            mask = F.interpolate(mask, self.img_size, mode='bicubic', align_corners=align_corners)
            mask = (mask > 0).type(torch.FloatTensor)
            # original_mask = F.interpolate(original_mask, self.img_size, mode='bicubic', align_corners=align_corners)
            # original_mask = (original_mask > 0).type(torch.FloatTensor)
            # original_mask = torch.squeeze(original_mask,0)
            img = torch.squeeze(img, 0)
            mask = torch.squeeze(mask,0)
            # These parameters are not used during training
            original_mask = 1
        

        return img, mask, original_mask, image_name, img_path

    def read_mask(self, path, mask_type):
        """
        Read Masks now only support bbox
        """
        if mask_type == 'random_bbox':
            bboxs = []
            for i in range(self.random_bbox_number):
                bbox = InpaintDataset.random_bbox(self.resize_shape, self.random_bbox_margin, self.random_bbox_shape)
                bboxs.append(bbox)
        elif mask_type == 'random_free_form':
            mask = InpaintDataset.random_ff_mask(self.random_ff_setting)
            return Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8))
        elif 'val' in mask_type:
            mask, original_mask = InpaintDataset.read_val_mask(path)
            return Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8)), Image.fromarray(np.tile(original_mask,(1,1,3)).astype(np.uint8))
        mask = InpaintDataset.bbox2mask(bboxs, self.resize_shape)
        return Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8))


    @staticmethod
    def read_val_mask(path):
        """
        Read masks from val mask data
        """
        if path.endswith("pkl"):
            mask = pkl.load(open(path, 'rb'))
        else:
            mask = Image.open(path)
            mask = np.array(mask).astype(np.uint8)
            # Keep it in range of [0,255], required for when using ToTensor()
            mask[mask > 128] = 255
            mask = np.invert(mask)
            mask[mask > 0] = 1
            original_mask = mask.copy()
            # Remove small holes in mask
            mask = ndimage.binary_dilation(mask).astype(mask.dtype)
            mask = ndimage.binary_dilation(mask).astype(mask.dtype)
            mask = ndimage.binary_erosion(mask).astype(mask.dtype)
            mask = ndimage.binary_erosion(mask).astype(mask.dtype)
            # Add images to keep information on he corner of the images.
            mask = original_mask + mask
            mask *= 255
            original_mask *= 255
            mask = np.expand_dims(mask, axis=2)
            original_mask = np.expand_dims(original_mask, axis=2)
        return mask, original_mask
    

    @staticmethod
    def read_bbox(path):
        """
        The general method for read bbox file by juding the file type
        Return:
            bbox:[y, x, height, width], shape: (height, width)
        """
        if filename[-3:] == 'pkl' and 'Human' in filename:
            return InpaintDataset.read_bbox_ch(filename)
        elif filename[-3:] == 'pkl' and 'COCO' in filename:
            return InpaintDataset.read_bbox_pkl(filename)
        else:
            return InpaintDataset.read_bbox_xml(path)

    @staticmethod
    def read_bbox_xml(path):
        """
        Read bbox for voc xml
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        with open(filename, 'r') as reader:
            xml = reader.read()
        soup = BeautifulSoup(xml, 'xml')
        size = {}
        for tag in soup.size:
            if tag.string != "\n":
                size[tag.name] = int(tag.string)
        objects = soup.find_all('object')
        bndboxs = []
        for obj in objects:
            bndbox = {}
            for tag in obj.bndbox:
                if tag.string != '\n':
                    bndbox[tag.name] = int(tag.string)

            bbox = [bndbox['ymin'], bndbox['xmin'], bndbox['ymax']-bndbox['ymin'], bndbox['xmax']-bndbox['xmin']]
            bndboxs.append(bbox)
        return bndboxs, (size['height'], size['width'])

    @staticmethod
    def read_bbox_pkl(path):
        """
        Read bbox from coco pkl
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        aux_dict = pkl.load(open(path, 'rb'))
        bbox = aux_dict["bbox"]
        shape = aux_dict["shape"]
        #bbox = random.choice(bbox)
        #fbox = bbox['fbox']
        return [[int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]], (shape[1], shape[0])

    @staticmethod
    def read_bbox_ch(path):
        """
        Read bbox from crowd human pkl
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        aux_dict = pkl.load(open(path, 'rb'))
        bboxs = aux_dict["bbox"]
        bbox = random.choice(bboxs)
        extra = bbox['extra']
        shape = aux_dict["shape"]
        while 'ignore' in extra and extra['ignore'] == 1 and bbox['fbox'][0] < 0 and bbox['fbox'][1] < 0:
            bbox = random.choice(bboxs)
            extra = bbox['extra']
        fbox = bbox['fbox']
        return [[fbox[1],fbox[0],fbox[3],fbox[2]]], (shape[1], shape[0])

    @staticmethod
    def read_seg_img(path):
        pass

    @staticmethod
    def random_bbox(shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)

        """
        img_height = shape[0]
        img_width = shape[1]
        height, width = bbox_shape
        ver_margin, hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)

    @staticmethod
    def random_ff_mask(config):
        """Generate a random free form mask with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)
        """

        h,w = config['img_shape']
        mask = np.zeros((h,w))
        num_v = 12+np.random.randint(config['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(config['ma'])
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10+np.random.randint(config['ml'])
                brush_w = 10+np.random.randint(config['mbw'])
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        mask*=255

        return mask.reshape(mask.shape+(1,)).astype(np.float32)


    @staticmethod
    def bbox2mask(bboxs, shape):
        """Generate mask tensor from bbox.

        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

        Returns:
            tf.Tensor: output with shape [1, H, W, 1]

        """
        height, width = shape
        mask = np.zeros(( height, width), np.float32)
        for bbox in bboxs:
            h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
            w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
            mask[bbox[0]+h:bbox[0]+bbox[2]-h,
                 bbox[1]+w:bbox[1]+bbox[3]-w] = 1.
        mask*=255
        return mask.reshape(mask.shape+(1,)).astype(np.float32)
