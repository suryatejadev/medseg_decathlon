import numpy as np
import os

from keras import backend, engine, layers, models, utils
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, Conv2D, Conv3D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D, Activation, Lambda
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def conv_block(x, n_filt, padding='same', dropout=0.0, batchnorm=True, dilation=(1,1)):
    def conv_l(inp):

        conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=dilation)(inp)
        conv = Activation('relu')(conv)
        conv = BatchNormalization()(conv) if batchnorm else conv
        conv = Dropout(dropout)(conv) if dropout>0.0 else conv
        return conv

    conv = conv_l(x)
    conv = conv_l(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv,pool

def upconv_block(x, x_conv, n_filt, padding='same', dropout=0.0, batchnorm=False):
    up_conv = Conv2DTranspose(n_filt, (2, 2), strides=(2, 2), padding=padding)(x)
    # crop x_conv
    if padding=='valid':
        ch, cw = get_crop_shape(x_conv, up_conv)
        x_conv = Cropping2D(cropping=(ch,cw), data_format="channels_last")(x_conv)
    up   = concatenate([up_conv, x_conv], axis=3)

    conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=(1,1))(up)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv) if batchnorm else conv
    conv = Dropout(dropout)(conv) if dropout>0.0 else conv

    conv = Conv2D(n_filt, (3, 3), padding=padding, dilation_rate=(1,1))(conv)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv) if batchnorm else conv
    conv = Dropout(dropout)(conv) if dropout>0.0 else conv
    return conv

def UNet2D(input_shape, features, depth, conv_params):

    # Define the input
    inputs = Input(input_shape)

    enc_conv = []
    current_input = inputs
    current_feat = features
    
    # Contracting path
    for i in range(depth):
        conv, pool = conv_block(current_input, current_feat, **conv_params)
        current_input = pool
        current_feat *= 2
        enc_conv.append(conv)

    # Expanding path
    _ = conv_params.pop('dilation')
    current_input = enc_conv.pop()
    current_feat /= 4
    for i in range(depth-1):
        conv = upconv_block(current_input, enc_conv.pop(), current_feat, **conv_params)
        current_input = conv
        current_feat /= 2
    conv = Conv2D(1, (1, 1), activation='sigmoid')(conv)
    
    return Model(inputs=[inputs], outputs=[conv])

