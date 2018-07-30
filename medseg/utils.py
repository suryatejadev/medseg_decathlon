import numpy as np
import nibabel as nib
import os
from skimage.transform import resize
from skimage.io import imread
from scipy.ndimage import affine_transform
import random
import math
from matplotlib import pyplot as plt
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import tifffile

def load_img(path):
    if '.nii.gz' in path:
        return nib.load(path).get_fdata()
    elif '.tif' in path:
        return tifffile.imread(path)
    elif '.npy' in path:
        return np.load(path)
    elif '.npz' in path:
        return np.load(path)['data']
    return imread(path, asgray=True)

def get_sublabel_value(sub_label, path):
    # Get task number: for Task02_Heart its 2
    task = int(path.split(os.sep)[2][5])
    # tasks = 7; total subclasses = 20
    # 1:4 , 2:2 , 3:3, 4:3, 5:3, 6:2 , 7:3
    task_vals = {
            1: [0, 13, 27, 40],
            2: [54, 67],
            3: [81, 94, 107],
            4: [121, 134, 148],
            5: [161, 174, 188],
            6: [201, 215],
            7: [228, 242, 255]
            }
    return task_vals[task][sub_label]


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

def init_session():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
