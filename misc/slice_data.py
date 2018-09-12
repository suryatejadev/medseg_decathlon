import numpy as np
import os
from glob import glob
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy.misc import imsave
from shutil import copyfile
from tqdm import tqdm

#  data_path = '/media/DATA/Datasets/medseg_decathlon/'
#  output_path = '/media/DATA/Datasets/medseg_decathlon/aux/data_sliced'
data_path = '../data/'
output_path = '../data_sliced/'

task_list = [x for x in os.listdir(data_path) if 'Task' in x]
for task in task_list:
    images_list =  os.listdir(data_path+task+'/imagesTr')

    task_folder = os.path.join(output_path, task)
    if os.path.exists(task_folder)==False:
        os.mkdir(task_folder)
        os.mkdir(os.path.join(task_folder, 'imagesTr'))
        os.mkdir(os.path.join(task_folder, 'labelsTr_npz'))
        #  copyfile(data_path+task+'/dataset.json', task_folder+'/dataset.json')
    print(task)
    for im_name in tqdm(images_list):
        if 'nii.gz' in im_name:
            # resample to isotropic
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
            # get slices
            depth = label3d.shape[2]
            for i in range(depth):
                img = img3d[...,i]
                label = label3d[...,i]
                imsave(task_folder+'/imagesTr/'+im_name[:-7]+'_'+str(i)+'.jpg', img)
                #  imsave(task_folder+'/labelsTr/'+im_name[:-7]+'_'+str(i)+'.jpg', label)
                np.savez_compressed(task_folder+'/labelsTr_npz/'+im_name[:-7]+'_'+str(i)+'.npz', data=label)
