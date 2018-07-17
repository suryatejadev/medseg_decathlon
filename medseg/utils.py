import numpy as np
import nibabel as nib
import os
from skimage.transform import resize
from scipy.ndimage import affine_transform
import random
import math


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

def augment_img(img, max_t=5, max_r=5, min_s=0.7, max_s=1.4):
	''' 
	Augments image by generating an affine matrix
	---------
	random translation betwen [-max_t,max_t], rotation between [-max_r, max_r],
	and scaling between [-min_s, max_s]
	
	'''
	def generate_perturbation_matrix_3D(max_t, max_r, min_s, max_s):
	    # Translation
	    tx = random.randint(-max_t,max_t + 1)
	    ty = random.randint(-max_t,max_t + 1)
	    tz = random.randint(-max_t,max_t + 1)
	    T = np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])
	    # Scaling
	    sx = np.random.uniform(min_s, max_s)
	    sy = np.random.uniform(min_s, max_s)
	    sz = np.random.uniform(min_s, max_s)
	    S = np.array([[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]])
	    # Rotation
	    rx = random.randint(-max_r,max_r+1) # x-roll (w.r.t x-axis)
	    ry = random.randint(-max_r,max_r+1) # y-roll
	    rz = random.randint(-max_r,max_r+1) # z-roll
	    c = math.cos(math.pi*rx/180)
	    s = math.sin(math.pi*rx/180)
	    Rx = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])
	    c = math.cos(math.pi*ry/180)
	    s = math.sin(math.pi*ry/180)
	    Ry = np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])
	    c = math.cos(math.pi*rz/180)
	    s = math.sin(math.pi*rz/180)
	    Rz = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
	    R = np.dot(np.dot(Ry,Rz), Rx)
	    #R = np.eye(4)
	    # Generate perturbation matrix
	    G = np.dot(np.dot(S,T), R)
	    #G = np.dot(S,R)
	    return G

	G = generate_perturbation_matrix_3D(max_t, max_r, min_s, max_s)
    G_mat = G[:3,:3]
    G_offset = G[:3,3]
    # perform affine trasform
    img_aug = affine_transform(img, G_mat, offset=G_offset, order=3)
    
    return img_aug
