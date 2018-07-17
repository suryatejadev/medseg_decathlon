import numpy as np
import nibabel as nib
import os
from skimage.transform import resize

def load_img(path):
    return nib.load(path).get_fdata()

def resize_img(img, img_dims):
    # Choose MR T2 channel for prostrate
    if len(img.shape)==4:
        img = img[...,0]
    return resize(img, img_dims[:-1], mode='reflect')[...,np.newaxis]

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)
