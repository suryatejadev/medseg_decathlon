import numpy as np
import os
import nibabel as nib
from shutil import copyfile
from tifffile import imsave
import sys
sys.path.append('..')

from medseg import utils

img_size = int(sys.argv[1])
data_path = '../data/'
output_path = '../data3d_size_'+str(img_size)
if os.path.exists(output_path)==False:
    os.mkdir(output_path)
for task in os.listdir(data_path):
    images_list =  os.listdir(data_path+task+'/imagesTr')

    task_folder = os.path.join(output_path, task)
    if os.path.exists(task_folder)==False:
        os.mkdir(task_folder)
        os.mkdir(os.path.join(task_folder, 'imagesTr'))
        os.mkdir(os.path.join(task_folder, 'labelsTr'))
        copyfile(data_path+task+'/dataset.json', task_folder+'/dataset.json')
    
    print(task)
    for im_name in images_list:
        if 'nii.gz' in im_name:
            img3d = nib.load(data_path+task+'/imagesTr/'+im_name).get_fdata()
            if len(img3d.shape)==4:
                img3d = img3d[...,0]
            label3d = nib.load(data_path+task+'/labelsTr/'+im_name).get_fdata()
            
            img3d_resize = utils.resize_img(img3d, (img_size, img_size, img_size))
            label3d_resize = utils.resize_img(label3d, (img_size, img_size, img_size))

            imsave(task_folder+'/imagesTr/'+im_name[:-7]+'.tif', img3d_resize)
            imsave(task_folder+'/labelsTr/'+im_name[:-7]+'.tif', label3d_resize)

