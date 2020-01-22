# data science
import numpy as np

# image
import matplotlib.pyplot as plt
from PIL import Image

# create the function
def fill_mask(row, col, height, width, orig_img):
    """
    Fill the mask cell with the nearest neighbour pixel of image
    Params:
      col, type(int)
      row, type(int)
      height, type(int)
      width, type(int)
    """
    # select a value from neighbour 
    fill_val_1 = 255
    fill_val_2 = 255
    fill_val_3 = 255
    if row-1 >= 0:
        fill_val_1 = orig_img[row-1][col][0]
        fill_val_2 = orig_img[row-1][col][1]
        fill_val_3 = orig_img[row-1][col][2]
    elif row+1 < height:
        fill_val_1 = orig_img[row+1][col][0]
        fill_val_2 = orig_img[row+1][col][1]
        fill_val_3 = orig_img[row+1][col][2]
    elif col-1 >= 0:
        fill_val_1 = orig_img[row][col-1][0]
        fill_val_2 = orig_img[row][col-1][1]
        fill_val_3 = orig_img[row][col-1][2]   
    elif col+1 < width:
        fill_val_1 = orig_img[row][col+1][0]
        fill_val_2 = orig_img[row][col+1][1]
        fill_val_3 = orig_img[row][col+1][2]  
    else:
        pass

    return fill_val_1, fill_val_2, fill_val_3



def detect_gradient(row, col, height, width, mask, orig_img):
    """
    Detects the whether the cell needs to be imputation or not
    """
    #print("Pos: %s, %s"%(row, col))
    mark_cell = False
    fill_val_1, fill_val_2, fill_val_3 = 0, 0, 0
    try:
        if (row-1) >= 0 and np.abs(mask[row-1][col] -  mask[row][col]) >= 200:
            mark_cell = True
            mark_row = row-1
            mark_col = col
        elif (row+1) < height and np.abs(mask[row+1][col] - mask[row][col]) >= 200:
            mark_cell = True
            mark_row = row+1
            mark_col = col
        elif (col-1) >= 0 and np.abs(mask[row][col-1] -  mask[row][col]) >= 200:
            mark_cell = True
            mark_row = row
            mark_col = col-1
        elif (col+1) < width and np.abs(mask[row][col+1] -  mask[row][col]) >= 200:
            mark_cell = True
            mark_row = row
            mark_col = col+1
        else:
            pass
    except:
        # Couldnt figured that bug out
        print("row+1", row)
        print("height", height)

    if mark_cell:
        #print(orig_img[mark_row][mark_col][0])
        fill_val_1 = orig_img[mark_row][mark_col][0]
        fill_val_2 = orig_img[mark_row][mark_col][1]
        fill_val_3 = orig_img[mark_row][mark_col][2]

    return mark_cell, fill_val_1, fill_val_2, fill_val_3


def shrink_mask(img, mask):
    """
    Shrinks mask via nearest neighbour imputation
    Params:
      img, type(PIL.JpegImagePlugin.JpegImageFile): 
      mask, type(PIL.JpegImagePlugin.JpegImageFile):
    Returns: 
      mask_new, type(PIL.Image.Image): Imputed mask 2D
    """
    width = img.size[1]
    height = img.size[0]
    print(type(width))
    # 3D array
    img_np = np.array(img, dtype='int64')
    # 2D array
    mask_np = np.array(mask, dtype='int64')
    # construct the new mask from original mask
    mask_new = np.dstack([np.array(mask), np.array(mask), np.array(mask)])

    for r in range(height):
        for c in range(width):
            mark_cell, fill_val_1, fill_val_2, fill_val_3 = detect_gradient(r, c, height, width, mask_np, img_np)
            if mark_cell:
                mask_new[r][c][0] = fill_val_1
                mask_new[r][c][1] = fill_val_2
                mask_new[r][c][2] = fill_val_3
    # numpy -> PIL.Image
    mask_new = Image.fromarray(mask_new, 'RGB')
    return mask_new 


