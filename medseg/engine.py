import numpy as np
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, CSVLogger

from medseg import models

class Model:
    def __init__(self, name, trained_model, compile_params, model_params):
        # Model architectures
        model_dict = {
                'Densenet3D': models.DenseNet3D
                }
        
        # Tensorflow initialize session
        self.init_session()
        
        # Load model architecture
        self.model = model_dict[name](**model_params)
        
        # Load saved weights
        if trained_model!='None':
            self.model.load_weights(trained_model)

        # Compile the model
        self.model.compile(**compile_params)

    def init_session(self):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
    
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

    def test(self, test_params):
        pass

