import numpy as np
import os 
import nibabel as nib
from glob import glob
import json

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
            img_batch.append(utils.resize_img(img, img_dims))
            labels_batch[i] = labels[batch_index[i]]
        img_batch = np.array(img_batch)

        yield(img_batch, labels_batch)

        

            



