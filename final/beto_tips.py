########### Reading Masks

    # Path is the path to the picture.
    def read_val_mask(path):
        mask = Image.open(path)
        mask = np.array(mask).astype(np.uint8)
        # Keep it in range of [0,255], required for when using ToTensor()
        # The mask is not all 0 or 255, it has values in the middle, but this works fine.
        mask[mask > 200] = 255
        mask = np.invert(mask)
        # mask = ndimage.binary_dilation(mask).astype(mask.dtype)
        mask = np.expand_dims(mask, axis=2)
        mas = Image.fromarray(np.tile(mask,(1,1,3)).astype(np.uint8))

        return mask


    # This part calls the above function.
    # After you get the mask, make the mask to a tensor
    # Something like
    self.transform_mask = transforms.Compose([
                           transforms.ToTensor(), # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           ])
    mask = self.transform_mask(read_val_mask('img.jpg'))

    # Then make the mask binary 0 or 1
    masks = (masks > 0).type(torch.FloatTensor)





############# Resizing Images
import torch.nn.functional as F
    align_corners=True
    img_size=512 # Or any other size
    # Shrinking the image to a square. # ori_imgs is the original image
    imgs = F.interpolate(imgs, ori_imgs, mode='bicubic', align_corners=align_corners)
    mask = F.interpolate(mask, ori_masks, mode='bicubic', align_corners=align_corners)

    # Making the image big again

    new_img = F.interpolate(imgs, (ori_imgs.shape[2], ori_imgs.shape[3]), mode='bicubic', align_corners=align_corners)
    # Change -1 and 1 to the expected minimum and maxiumum value of your image. 
    # Could be 0 to 1, or 0 to 255, depends on your case, but you need to clip it
    new_img = new_img.clamp(min=-1, max=1)

