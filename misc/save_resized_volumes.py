import numpy as np
import os
import nibabel as nib
from shutil import copyfile
from nibabel.processing import resample_to_output
from tifffile import imsave
import sys
sys.path.append('..')

from medseg import utils

img_size = int(sys.argv[1])
data_path = '/media/DATA/Datasets/medseg_decathlon/'
output_path = '/media/DATA/Datasets/medseg_decathlon/aux/data3d_size_'+str(img_size)
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
            img3d_nib = nib.load(data_path+task+'/imagesTr/'+im_name)
            img3d = img3d_nib.get_fdata()
            if len(img3d.shape)==4:
                img3d = img3d[...,0]
                img3d_nib = nib.Nifti1Image(img3d, img3d_nib.affine, img3d_nib.header)
            img3d_nib = resample_to_output(img3d_nib, voxel_sizes=1, order=2)
            img3d = img3d_nib.get_fdata()

            label3d_nib = nib.load(data_path+task+'/labelsTr/'+im_name)
            label3d_nib = resample_to_output(label3d_nib, voxel_sizes=1, order=0)
            label3d = label3d_nib.get_fdata()

            img3d_resize = utils.resize_img(img3d, (img_size, img_size, img_size))
            label3d_resize = utils.resize_img(label3d, (img_size, img_size, img_size))

            np.savez_compressed(task_folder+'/imagesTr/'+im_name[:-7]+'.npz', data=img3d_resize)
            np.savez_compressed(task_folder+'/labelsTr/'+im_name[:-7]+'.npz', data=label3d_resize)
