import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.losses import categorical_crossentropy
import keras.backend as K

from medseg import models

class Model:
    def __init__(self, name, trained_model, compile_params, model_params):
        # Model architectures
        model_dict = {
                'DenseNet3D': models.DenseNet3D,
                'UNet2D': models.UNet2D,
                'DilatedDenseNet': models.DilatedDenseNet
                }
        
        # Load model architecture
        self.model = model_dict[name](**model_params)
        print(self.model.summary())
        
        # Load saved weights
        if trained_model!='None':
            self.model.load_weights(trained_model)

        # Compile the model
        self.model.compile(
                optimizer=compile_params['optimizer'],
                loss=self.loss(compile_params['loss']),
                metrics=compile_params['metrics'])

    def loss(self, loss_name):
        if loss_name=='categorical_crossentropy':
            loss = categorical_crossentropy
        elif loss_name=='dice_coef':
            def dice_coef(y_true, y_pred, smooth=1):
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(y_pred)
                intersection = K.sum(y_true_f * y_pred_f)
                dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
                return 1 - dice_coef
            loss = dice_coef
        return loss

    def train(self, datagen_train, datagen_val, output_dir, ckpt_period, fit_params):
        # Callbacks
        wt_path = output_dir+'/checkpoints/'+'wt-{epoch:02d}-{val_acc:.2f}.h5'
        wt_saver = ModelCheckpoint(wt_path, monitor='val_acc', verbose=1,
                                     save_weights_only=True, period=ckpt_period)
        log_path = output_dir+'/logs/log.csv'
        csv_logger = CSVLogger(log_path, separator=',', append=False)
        callbacks = [wt_saver, csv_logger]
        
        # Train the model
        self.model.fit_generator(generator=datagen_train, 
                validation_data=datagen_val, callbacks=callbacks,
                verbose=2, **fit_params)

        # Save the plots
        self.save_plots(log_path, output_dir+'/logs/')

    def save_plots(self, log_path, output_dir):
        # Save the plots
        logs = pd.read_csv(log_path)
        acc_train, loss_train = logs['acc'], logs['loss']
        acc_val, loss_val = logs['val_acc'], logs['val_loss']
        # Plot accuracy
        plt.figure()
        plt.plot(acc_train, c='b', label='Training')
        plt.plot(acc_val, c='g', label='Validation')
        plt.title('Accuracy'); plt.legend();
        plt.savefig(output_dir+'accuracy.png', dpi=300); plt.close()
        # Plot loss
        plt.figure()
        plt.plot(loss_train, c='b', label='Training')
        plt.plot(loss_val, c='g', label='Validation')
        plt.title('Loss'); plt.legend();
        plt.savefig(output_dir+'loss.png', dpi=300); plt.close()

    def test(self, test_params):
        pass

