import numpy as np
import os 
import nibabel as nib
from glob import glob
import json
import random
import math
from scipy.ndimage import affine_transform

from keras.preprocessing.image import apply_affine_transform
from medseg import utils

# Get the file paths for train and validation images and labels
def get_paths(data_path, validation_split=0.2):
    task_list = [x for x in os.listdir(data_path) if x.startswith('Task')]
    N_tasks = len(task_list)

    num_labels = []
    img_paths = {}
    annot_paths = {}
    
    for i, task in enumerate(task_list):
        # Load the data.json file
        with open(os.path.join(data_path,task,'dataset.json')) as f:
            task_info = json.load(f)
        # Number of labels in each task
        num_labels.append(len(task_info['labels']))
    total_labels = np.sum(num_labels)

    # Get train and validation datapaths and labels
    paths_train, paths_val = [], []
    labels_train, labels_val = [], []
    annot_train, annot_val = [], []
    for i, task in enumerate(task_list):
        # Image names in the task
        img_path = os.path.join(data_path,task,'imagesTr')
        img_names = os.listdir(img_path)
        num_labels_task = num_labels[i]
        
        num_images = len(img_names)
        num_val = int(np.floor(validation_split*num_images))
        num_train = num_images - num_val
        
        # Get list of train images
        img_paths_task = []
        for name in img_names:
            img_paths_task.append(os.path.join(img_path, name))
        paths_train.append(img_paths_task[:num_train])
        paths_val.append(img_paths_task[num_train:])       

        # Get list of annotation images
        annot_path = os.path.join(data_path,task,'labelsTr')
        annot_paths_task = []
        for name in img_names:
            annot_paths_task.append(os.path.join(annot_path, name))
        annot_train.append(annot_paths_task[:num_train])
        annot_val.append(annot_paths_task[num_train:])
        
        # Get list of labels
        label_task = np.zeros((total_labels))
        start_index = int(np.sum(num_labels[:i]))
        end_index = start_index + num_labels[i]
        label_task[start_index:end_index] = 1

        labels_train.append(np.array([label_task]*num_train))
        labels_val.append(np.array([label_task]*num_val))
    
    paths_train = sum(paths_train, [])
    paths_val = sum(paths_val, [])

    annot_train = sum(annot_train, [])
    annot_val = sum(annot_val, [])
 
    labels_train = np.concatenate(tuple(labels_train))
    labels_val = np.concatenate(tuple(labels_val))

    return paths_train, annot_train, labels_train, paths_val, annot_val, labels_val

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

def augment_img2d(img, tx_params=None, max_r=5, max_t=5, max_s=1.4):
    if tx_params==None:
        tx_params = {}
        tx_params['theta'] = np.random.randint(-max_r, max_r+1)
        tx_params['tx'] = np.random.randint(-max_t, max_t+1)
        tx_params['ty'] = np.random.randint(-max_t, max_t+1)
        tx_params['zx'] = np.random.uniform(-max_s, max_s)
        tx_params['zy'] = np.random.uniform(-max_s, max_s)

    img_tx = apply_affine_transform(img, **tx_params)
    return img_tx, tx_params

def datagen_classify(data, labels, batch_size, img_dims):
    while True:
        # Get batch indices from uniformly random distribution
        N = len(data)
        batch_index = np.arange(N)
        np.random.shuffle(batch_index)
        batch_index = batch_index[:batch_size]

        # Get batch data
        img_batch = []
        labels_batch = np.zeros((batch_size, labels.shape[1]))
        for i in range(batch_size):
            img = utils.load_img(data[batch_index[i]])
            img = utils.resize_img(img, np.array(img_dims)[:-1])
            img = utils.normalize(img, maxval=255)
            img = augment_img(img)[...,np.newaxis]
            img = utils.normalize(img, maxval=1)
            img_batch.append(img)
            labels_batch[i] = labels[batch_index[i]]
        img_batch = np.array(img_batch)

        yield(img_batch, labels_batch)

def datagen_segment(data, annots, batch_size, img_dims):
    while True:
        # Get batch indices from uniformly random distribution
        N = len(data)
        batch_index = np.arange(N)
        np.random.shuffle(batch_index)
        batch_index = batch_index[:batch_size]

        # Get batch data
        img_batch = []
        annot_batch = []
        for i in range(batch_size):
            # Get image
            img = utils.load_img(data[batch_index[i]])
            img = utils.resize_img(img, img_dims)
            img, tx_params = augment_img2d(img)
            img_batch.append(img)
            
            # Get annotation
            annot = utils.load_img(annots[batch_index[i]])
            annot[np.where(annot>0)] = 255
            annot = utils.resize_img(annot, img_dims)
            annot,_ = augment_img2d(annot, tx_params)
            annot_batch.append(annot)
            
        img_batch = np.array(img_batch)
        annot_batch = np.array(annot_batch)
        yield(img_batch, annot_batch)

