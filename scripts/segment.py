from __future__ import print_function
import sys
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile

from medseg import datagen, engine, utils

def segment(config_file):
    
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    # Create experiment output directories
    exp_dir = os.path.join(config['output_dir'], config['exp_id'])
    utils.create_dirs([exp_dir,
            os.path.join(exp_dir, 'checkpoints'),
            os.path.join(exp_dir, 'logs')])

    # Copy config file
    copyfile(config_file, os.path.join(exp_dir, config['exp_id']+'.yaml'))

    # Train and validation data generators
    paths_train, annot_train, _, paths_val, annot_val, _\
            = datagen.get_paths(config['data_path'])
    datagen_train = datagen.datagen_segment(paths_train,
            annot_train, **config['datagen'])
    datagen_val = datagen.datagen_segment(paths_val,
            annot_val, **config['datagen'])
   
    x,y = next(datagen_train)
    print(x.shape, x.min(), x.max(), y.shape, y.min(), y.max())
    return x,y
    
    # Tensorflow initialize session
    #  utils.init_session()
    
    # Define Classification model
    #  model = engine.Model(**config['model'])

    #  # Train the model
    #  model.train(datagen_train, datagen_val, exp_dir, **config['train'])

    # Test the model
    #  model.test(**config['test'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    x,y = segment(args.config)
    

