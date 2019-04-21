#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:20:35 2019

@author: muzhi
"""

from keras.layers import Input, Dense, Flatten
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation


def simple_net(inputs):
    x = Dense(64, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    return x

def conv_shared(inputs):
    x = inputs
    for _ in range(2):
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = Conv2D(1, kernel_size=(1, 1), padding="same", activation=None)(x)
    x = Flatten()(x)
    x = Activation('relu')(x)
    return x

def conv_breakout(inputs):
    x = inputs
    for _ in range(4):
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
    x = Activation('relu')(x)
    x = Conv2D(1, kernel_size=(1, 1), padding="same", activation=None)(x)
    x = Flatten()(x)
    x = Activation('relu')(x)
    x = Dense(128, activation="relu")(x)
    return x
