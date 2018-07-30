import numpy as np
import nibabel as nib
import os
from skimage.transform import resize
from skimage.io import imread
from scipy.ndimage import affine_transform
import scipy
import random
import math
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
    task = int(path.split(os.sep)[-3][5])
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


def resize_img(img, img_dims, order=3):
    # Choose MR T2 channel for prostrate
    if len(img.shape)==4:
        img = img[...,0]
    return resize(img, img_dims, order=order, mode='reflect')

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

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0., horizontal_flip=False):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)

        if horizontal_flip:
            x = np.fliplr(x)
    return x
