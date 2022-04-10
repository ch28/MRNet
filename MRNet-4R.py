#-----------------------------------------------------------------------------------+
# Author: Xin Liang                                                                 |
# Time Stamp: Oct 1, 2021                                                           |
# Affiliation: Beijing University of Posts and Telecommunications                   |
# Email: liangxin@bupt.edu.cn                                                       |
#-----------------------------------------------------------------------------------+
#                             *** Open Source Code ***                              |
#-----------------------------------------------------------------------------------+
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, Lambda, Conv3D, concatenate
from keras.models import Model
from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
import scipy.io as sio 
import numpy as np
import math
import time
import hdf5storage # load Matlab data bigger than 2GB

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.reset_default_graph()

# 40%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)

envir = 'indoor' # 'indoor' or 'outdoor'

# training params
epochs = 1000
batch_size = 50

# image params
img_height = 32
img_width = 32
img_depth = 4 # num of Rx
img_channels = 2 
img_total = img_height*img_width*img_depth*img_channels

residual_num = 2
encoded_dim = 512  # compress rate=1/4->dim.=512, compress rate=1/16->dim.=128

# Bulid the autoencoder model of MRNet-4R
def residual_network(x, residual_num, encoded_dim):
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y
    def residual_block_decoded(y):
        shortcut = y
        y = Conv3D(8, kernel_size=(3, 3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv3D(16, kernel_size=(3, 3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv3D(16, kernel_size=(3, 3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y

    # layer reuse
    dense_shared_enc = Dense(encoded_dim, activation='linear')
    dense_shared_dec = Dense(img_height*img_width*img_channels, activation='linear')
    # encoder
    x = Conv3D(16, (3, 3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x = Conv3D(2, (3, 3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    x1 = Lambda(lambda x: x[:, :, 0:1, :, :])(x)
    x2 = Lambda(lambda x: x[:, :, 1:2, :, :])(x)
    x3 = Lambda(lambda x: x[:, :, 2:3, :, :])(x)
    x4 = Lambda(lambda x: x[:, :, 3:4, :, :])(x)

    x1 = Reshape((img_height*img_width*img_channels,))(x1)
    x2 = Reshape((img_height*img_width*img_channels,))(x2)
    x3 = Reshape((img_height*img_width*img_channels,))(x3)
    x4 = Reshape((img_height*img_width*img_channels,))(x4)

    encoded1 = dense_shared_enc(x1) # encoded result for Rx1
    encoded2 = dense_shared_enc(x2) # encoded result for Rx2
    encoded3 = dense_shared_enc(x3) # encoded result for Rx3
    encoded4 = dense_shared_enc(x4) # encoded result for Rx4
    
    # decoder
    x1 = dense_shared_dec(encoded1)
    x2 = dense_shared_dec(encoded2)
    x3 = dense_shared_dec(encoded3)
    x4 = dense_shared_dec(encoded4)

    x1 = Reshape((img_channels, 1, img_height, img_width,),)(x1)
    x2 = Reshape((img_channels, 1, img_height, img_width,),)(x2)
    x3 = Reshape((img_channels, 1, img_height, img_width,),)(x3)
    x4 = Reshape((img_channels, 1, img_height, img_width,),)(x4)
        
    x = concatenate([x1, x2, x3, x4], axis = 2)

    x = Conv3D(16, (3, 3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)
    
    x = Conv3D(2, (3, 3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x

image_tensor = Input(shape=(img_channels, img_depth, img_height, img_width))
network_output = residual_network(image_tensor, residual_num, encoded_dim)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())


# Data loading
dataset_path = './data'

if envir == 'indoor':
    data_path = dataset_path + '/Hdata_indoor_4Rx.mat'
    mat = hdf5storage.loadmat(data_path)
    x_train = mat['T1_train_norm'] # array
    x_val = mat['T1_val_norm'] # array
    x_test = mat['T1_test_norm'] # array

elif envir == 'outdoor':
    data_path = dataset_path + '/Hdata_outdoor_4Rx.mat'
    mat = hdf5storage.loadmat(data_path)
    x_train = mat['T1_train_norm'] # array    
    x_val = mat['T1_val_norm'] # array    
    x_test = mat['T1_test_norm'] # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

x_train = np.reshape(x_train, (len(x_train), img_channels, img_depth, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (len(x_val), img_channels, img_depth, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_channels, img_depth, img_height, img_width))  # adapt this if using `channels_first` image data format

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        

file = 'MRNet-4R_'+(envir)+'_dim'+str(encoded_dim)
path = 'result/TensorBoard_%s' %file

save_dir = os.path.join(os.getcwd(), 'result/')
model_name = '%s_model.h5' % file
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)


history = LossHistory()

callbacks = [history, TensorBoard(log_dir = path)]

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=callbacks)


outfile = 'result/%s_model.h5' % file
autoencoder.save_weights(outfile)

# Testing data
autoencoder.load_weights(outfile)

x_hat = autoencoder.predict(x_test)

# Rx1
x_test_real = np.reshape(x_test[:, 0, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, 0, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, 0, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
nmse1 = np.mean(mse/power)

# Rx2
x_test_real = np.reshape(x_test[:, 0, 1, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, 1, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
nmse2 = np.mean(mse/power)


# Rx3
x_test_real = np.reshape(x_test[:, 0, 2, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, 2, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, 2, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, 2, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
nmse3 = np.mean(mse/power)


# Rx4
x_test_real = np.reshape(x_test[:, 0, 3, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, 3, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, 3, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, 3, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)

power = np.sum(abs(x_test_C)**2, axis=1)

mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)
nmse4 = np.mean(mse/power)

nmse = (nmse1 + nmse2 + nmse3 + nmse4) / 4

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(nmse))















