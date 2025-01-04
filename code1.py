import os
from keras import Model
from numpy.random import randint
import keras
import tensorflow as tf
from keras import initializers
from os import listdir
import numpy as np
from numpy import asarray, vstack, savez_compressed, load, ones, zeros
from keras.preprocessing.image import img_to_array, load_img
import cv2
import keras.optimizers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Activation, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


def load_images(path, size=(256,256)):
    data_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        data_list.append(pixels)
    return asarray(data_list)

path_train = 'mini_training_set/'
path_test = 'mini_test_set/'

data_blurred1 = load_images(path_train + 'blurred/')
data_blurred2 = load_images(path_test + 'blurred/')
data_blurred = vstack((data_blurred1, data_blurred2))
print('loaded blurred data :', data_blurred.shape)

data_normal1 = load_images(path_train + 'normal/')
data_normal2 = load_images(path_test + 'normal/')
data_normal = vstack((data_normal1, data_normal2))
print('loaded normal data:', data_normal.shape)

filename = 'mini_blurred_to_normal3.npz'
savez_compressed(filename, data_blurred, data_normal)
print('saved dataset:', filename) #(468, 468)


data = load('mini_blurred_to_normal3.npz')
data_blurred3, data_normal3 = data['arr_0'], data['arr_1']
print('loaded:', data_blurred3.shape, data_normal3.shape)

n_samples = 3
for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(data_blurred3[i].astype('uint8'))

for i in range(n_samples):
    plt.subplot(2, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(data_normal3[i].astype('uint8'))

plt.show()
