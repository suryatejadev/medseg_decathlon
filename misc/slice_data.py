import numpy as np
import os
from glob import glob
import nibabel as nib
from scipy.misc import imsave
from tqdm import tqdm
from shutil import copyfile

data_path = '../data/'
output_path = '../data_sliced'
for task in os.listdir(data_path):
    images_list =  os.listdir(data_path+task+'/imagesTr')

    task_folder = os.path.join(output_path, task)
    if os.path.exists(task_folder)==False:
        os.mkdir(task_folder)
        os.mkdir(os.path.join(task_folder, 'imagesTr'))
        os.mkdir(os.path.join(task_folder, 'labelsTr_npy'))
        copyfile(data_path+task+'/dataset.json', task_folder+'/dataset.json')
    
    print(task)
    for im_name in images_list:
        if 'nii.gz' in im_name:
            img3d = nib.load(data_path+task+'/imagesTr/'+im_name).get_fdata()
            if len(img3d.shape)==4:
                img3d = img3d[...,0]
            label3d = nib.load(data_path+task+'/labelsTr/'+im_name).get_fdata()
            depth = label3d.shape[2]
            for i in range(depth):
                img = img3d[...,i]
                label = label3d[...,i]
                imsave(task_folder+'/imagesTr/'+im_name[:-7]+'_'+str(i)+'.jpg', img)
                #  imsave(task_folder+'/labelsTr/'+im_name[:-7]+'_'+str(i)+'.jpg', label)
                np.save(task_folder+'/labelsTr_npy/'+im_name[:-7]+'_'+str(i)+'.npy', label)


