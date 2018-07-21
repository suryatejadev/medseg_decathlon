import numpy as np
import os
from keras import layers, backend, models

def dense_block3D(x, blocks, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block3D(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block3D(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 4
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv3D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling3D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block3D(x, growth_rate, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 4
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv3D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv3D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def DenseNet3D(blocks,
             input_shape=None,
             classes=1000):
    img_input = layers.Input(shape=input_shape)
    bn_axis = 4
    
    # Conv Layer 1
    x = layers.ZeroPadding3D(padding=((3, 3), (3, 3), (3,3)))(img_input)
    x = layers.Conv3D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding3D(padding=((1, 1), (1, 1), (1, 1)))(x)
    x = layers.MaxPooling3D(3, strides=2, name='pool1')(x)
    
    # Dense Blocks
    for i, block in enumerate(blocks):
        x = dense_block3D(x, block, name='conv'+str(i+2))
        if i<len(blocks)-1:
            x = transition_block3D(x, 0.5, name='pool'+str(i+2))

    # Final Layers
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc')(x)
    
    # Create model
    model = models.Model(img_input, x, name='densenet3D')
    return model

