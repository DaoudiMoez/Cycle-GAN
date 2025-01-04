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

# def load_images(path, size=(256,256)):
#     data_list = list()
#     for filename in listdir(path):
#         pixels = load_img(path + filename, target_size=size)
#         pixels = img_to_array(pixels)
#         data_list.append(pixels)
#     return asarray(data_list)
#
# path_train = 'mini_training_set/'
# path_test = 'mini_test_set/'

# defining optimizer
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate= 0.0003, beta_1=0.5)

# Define the discriminator model
def define_discriminator(image_shape):
    init = initializers.RandomNormal(stddev=0.02)

    # Input layer
    in_image = Input(shape=image_shape)

    # Add convolutional layers
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Output patch prediction
    patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)

    # Compile the model
    model = Model(in_image, patch_out)
    model.compile(loss='mse', optimizer=optimizer, loss_weights=[0.5])
    return model

# Define a ResNet block
def resnet_block(n_filters, input_layer):
    init = initializers.RandomNormal(stddev=0.02)

    # First convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # Second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    # Concatenate input and output
    g = Concatenate()([g, input_layer])
    return g

# Define the generator model
def define_generator(image_shape, n_resnet=9):
    init = initializers.RandomNormal(stddev=0.02)

    # Input layer
    in_image = Input(shape=image_shape)

    # Initial convolutional layer
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # Downsampling layers
    g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # ResNet blocks
    for _ in range(n_resnet):
        g = resnet_block(256, g)

    # Upsampling layers
    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # Output layer
    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)

    # Define model
    model = Model(in_image, out_image)
    return model


# Load real samples from file
def load_real_samples(filename):
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# Generate real samples
def generate_real_samples(dataset, n_samples, patch_shape):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# Generate fake samples
def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# Save generator model to file
def save_model(step, g_model):
    filename = 'g_model_AtoB_%06d.h5' % (step + 1)
    g_model.save(filename)
    print('>Saved: %s' % filename)

# Summarize performance
def summarize_performance(step, g_model, trainA, n_samples=5):
    X_in, _ = generate_real_samples(trainA, n_samples, 0)
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0

    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        plt.axis('off')
        plt.imshow(X_in[i])

    for i in range(n_samples):
        plt.subplot(2, n_samples, n_samples + i + 1)
        plt.axis('off')
        plt.imshow(X_out[i])

    filename1 = 'AtoB_generated_plot_%06d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()


def update_image_pool(pool, images, max_size = 50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# Train CycleGAN models
def train(d_model, g_model, dataset):
    n_epochs, n_batch = 100, 1
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        # Generate real samples
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

        # Generate fake samples
        X_fakeB, y_fakeB = generate_fake_samples(g_model, X_realA, n_patch)

        # Update discriminator
        d_loss1 = d_model.train_on_batch(X_realB, y_realB)
        d_loss2 = d_model.train_on_batch(X_fakeB, y_fakeB)

        # Update generator
        g_loss = g_model.train_on_batch(X_realA, y_realB)

        print('>%d, d[%.3f,%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))

        # Summarize performance
        if (i + 1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, trainA)

        # Save model
        if (i + 1) % (bat_per_epo * 5) == 0:
            save_model(i, g_model)

# Load dataset
dataset = load_real_samples('mini_blurred_to_normal.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)

# Define image shape
image_shape = dataset[0].shape[1:]

# Define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

# Train the model
train(d_model, g_model, dataset)
