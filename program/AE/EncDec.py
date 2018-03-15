import numpy as np
import sys,os
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
from scipy import stats
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras.layers as kadd
from tqdm import tqdm
from keras.callbacks import TensorBoard

def build_encoder(img_shape):
    input_img = Input(shape=(img_shape)) 
    x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(540, activation='relu')(x)
    Encoder=Model(input_img,encoded,name='encoder')
    return input_img,encoded,Encoder
def build_decoder(encoded):
    input_code=Input(shape=encoded.get_shape().as_list()[1:])
    x = Reshape((720//48,576//48,3))(input_code)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    Decoder=Model(input_code,decoded,name='decoder')
    return input_code,decoded,Decoder
def build_AE(e,d):
    AE=Sequential()
    AE.add(e)
    AE.add(d)
    AE.compile(optimizer='adam', loss='mse')
    return AE