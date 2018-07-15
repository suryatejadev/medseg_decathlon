import sys
sys.path.append('..')
import os
import yaml
import argparse
from shutil import copyfile

from medseg import engine, datagen, utils

def classify(config_file):
    # Parser config file
    with open(config_file) as f:
        config = yaml.load(f)

    # Create experiment output directories

    # Copy config file

    # Define Classification model
    model = engine.Model(**config['model'])
    
    # Train and test data generators
    datagen_train = datagen.datagen(**config['datagen_train'])
    datagen_test = datagen.datagen(**config['datagen_test'])

    # Train the model
    model.train(datagen_train, datagen_test, **config['train'])

    # Test the model
    model.test(**config['test'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.args()
    classify(args.config)
