import numpy as np
import nibabel as nib
import os
from skimage.transform import resize
from scipy.ndimage import affine_transform
import random
import math
import cv2

def load_img(path):
    if '.nii.gz' in path:
        return nib.load(path).get_fdata()
    return cv2.imread(path, 0)

def resize_img(img, img_dims):
    # Choose MR T2 channel for prostrate
    if len(img.shape)==4:
        img = img[...,0]
    return resize(img, img_dims, mode='reflect')

def normalize(img, maxval=1):
    img_min = img.min()
    img_max = img.max()
    return maxval*(img-img_min)/(img_max-img_min)

def create_dirs(paths):
    for path in paths:
        if os.path.exists(path)==False:
            os.mkdir(path)


