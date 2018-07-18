import numpy as np
import os 
import nibabel as nib
from glob import glob
import json
import random
import math
from scipy.ndimage import affine_transform

from medseg import utils

# Get the file paths for train and validation images and labels
def get_paths(data_path, validation_split=0.2):
    task_list = os.listdir(data_path)
    N_tasks = len(task_list)

    num_labels = []
    img_paths = {}
    label_paths = {}
    
    for i, task in enumerate(task_list):
        # Load the data.json file
        with open(os.path.join(data_path,task,'dataset.json')) as f:
            task_info = json.load(f)
        
        # Number of labels in each task
        num_labels.append(len(task_info['labels']))
        
        # Get image and label paths in each task
        img_paths_task = []
        label_paths_task = []
        for path in task_info['training']:
            img_paths_task.append(os.path.join(data_path, task, path['image']))
            label_paths_task.append(os.path.join(data_path, task, path['label']))
        img_paths[task] = img_paths_task
        label_paths[task] = label_paths_task

    total_labels = np.sum(num_labels)

    # Get train and validation datapaths and labels
    paths_train, paths_val = [], []
    labels_train, labels_val = [], []
    for i, task in enumerate(task_list):
        img_paths_task = img_paths[task]
        num_labels_task = num_labels[i]
        
        num_images = len(img_paths_task)
        num_val = int(np.floor(validation_split*num_images))
        num_train = num_images - num_val

        paths_train.append(img_paths_task[:num_train])
        paths_val.append(img_paths_task[num_train:])
        
        label_task = np.zeros((total_labels))
        start_index = int(np.sum(num_labels[:i]))
        end_index = start_index + num_labels[i]
        label_task[start_index:end_index] = 1

        labels_train.append(np.array([label_task]*num_train))
        labels_val.append(np.array([label_task]*num_val))
    
    paths_train = sum(paths_train, [])
    paths_val = sum(paths_val, [])
    
    labels_train = np.concatenate(tuple(labels_train))
    labels_val = np.concatenate(tuple(labels_val))

    return paths_train, labels_train, paths_val, labels_val

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

def datagen(data, labels, batch_size, img_dims):
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
            img = utils.resize_img(img, img_dims)
            img = utils.normalize(img, maxval=255)
            img = augment_img(img)[...,np.newaxis]
            img = utils.normalize(img, maxval=1)
            img_batch.append(img)
            labels_batch[i] = labels[batch_index[i]]
        img_batch = np.array(img_batch)

        yield(img_batch, labels_batch)

        

            



