from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, Dropout, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Model
from keras import metrics
from keras import backend as K

import pickle
import h5py

import numpy as np
import gzip

from time import time

gan_path = ""


# muenster_path = gan_path + "muenster_AE.pkl.gz"
# read data - for AE only X_train, x_test is needed
hf = h5py.File('X_train.h5', 'r')
X_train = np.zeros((54000, 180, 180, 1), dtype='float32')
hf['X_train'].read_direct(X_train)
hf.close()
hf = h5py.File('x_test.h5', 'r')
x_test = np.zeros((6000, 180, 180, 1), dtype='float32')
hf['x_test'].read_direct(x_test)
hf.close()
# hf = h5py.File('B_train.h5', 'r')
# y_train = np.zeros((54000, 1), dtype='float32')
# hf['B_train'].read_direct(y_train)
# hf.close()
# hf = h5py.File('b_test.h5', 'r')
# y_test = np.zeros((6000, 1), dtype='float32')
# hf['b_test'].read_direct(y_test)
# hf.close()
# X_train = X_train[:540,:,:,:]
# x_test = x_test[:60,:,:,:]
print(X_train.shape, x_test.shape)

img_rows, img_cols, img_chns = 180, 180, 1
original_img_size = (img_rows, img_cols, img_chns)

batch_size = 50
latent_dim = 256
# intermediate_dim = 512
epsilon_std = 1.0
epochs = 50
activation = 'relu'
dropout = 0.5
learning_rate = 0.0001
decay = 0.0

# Muenster Data is already normalized
# f = gzip.open(muenster_path, 'rb')
# [X_train, X_val, X_test, y_train, y_val, y_test] = pickle.load(f)
# f.close()
# X_train  = np.array(X_train)
# X_val  = np.array(X_val)
# X_test  = np.array(X_test)
# y_train  = np.array(y_train)
# y_val  = np.array(y_val)
# y_test  = np.array(y_test)

# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
# print(X_test.shape, y_test.shape)

# Synthetic data is already normalized


def create_enc_conv_layers(stage, **kwargs):
    conv_name = '_'.join(['enc_conv', str(stage)])
    bn_name = '_'.join(['enc_bn', str(stage)])
    layers = [
        Conv2D(name=conv_name, **kwargs),
        BatchNormalization(name=bn_name),
        Activation(activation),
    ]
    return layers


def create_dense_layers(stage, width):
    dense_name = '_'.join(['enc_dense', str(stage)])
    bn_name = '_'.join(['enc_bn', str(stage)])
    layers = [
        Dense(width, name=dense_name),
        BatchNormalization(name=bn_name),
        Activation(activation),
        Dropout(dropout),
    ]
    return layers


def inst_layers(layers, in_layer):
    x = in_layer
    for layer in layers:
        if isinstance(layer, list):
            x = inst_layers(layer, x)
        else:
            x = layer(x)

    return x

enc_filters=64
enc_layers = [
    create_enc_conv_layers(stage=1, filters=enc_filters, kernel_size=5, strides=2, padding='same'),
    create_enc_conv_layers(stage=2, filters=enc_filters, kernel_size=5, strides=2, padding='same'),
    create_enc_conv_layers(stage=3, filters=enc_filters, kernel_size=5, strides=1, padding='same'),
    Flatten(),
    # create_dense_layers(stage=4, width=intermediate_dim),
]

x = Input(batch_shape=(batch_size,) + original_img_size)
_enc_dense = inst_layers(enc_layers, x) # private variables

_z_mean_1 = Dense(latent_dim)(_enc_dense)
_z_log_var_1 = Dense(latent_dim)(_enc_dense)

z_mean = _z_mean_1
z_log_var = _z_log_var_1


def sampling(args, batch_size=batch_size, latent_dim=latent_dim, epsilon_std=epsilon_std):
    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)

    return z_mean + K.sqrt(K.exp(z_log_var)) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

def create_dec_trans_conv_layers(stage, **kwargs):
    conv_name = '_'.join(['dec_trans_conv', str(stage)])
    bn_name = '_'.join(['dec_bn', str(stage)])
    layers = [
        Conv2DTranspose(name=conv_name, **kwargs),
        BatchNormalization(name=bn_name),
        Activation(activation),
    ]
    return layers
dec_filters = 64
decoder_layers = [
    create_dense_layers(stage=10, width=45 * 45 * enc_filters),
    Reshape((45, 45, enc_filters)),
    create_dec_trans_conv_layers(11, filters=dec_filters, kernel_size=5, strides=1, padding='same'),
    create_dec_trans_conv_layers(12, filters=dec_filters, kernel_size=5, strides=2, padding='same'),
    create_dec_trans_conv_layers(13, filters=dec_filters, kernel_size=5, strides=2, padding='same'),
    Conv2DTranspose(name='x_decoded', filters=img_chns, kernel_size=1, strides=1, activation='sigmoid'),
]

_dec_out = inst_layers(decoder_layers, z)
_output = _dec_out


def kl_loss(x, x_decoded_mean):
    # KL divergence approximation in Gaussian case
    kl_loss = - 0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    return K.mean(kl_loss)


def logx_loss(x, x_decoded_mean):
    # Reconstruction Loss - binary_crossentropy compares whether the pixel in the decoded image is the same as the original image
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * img_chns * metrics.binary_crossentropy(x, x_decoded_mean)
    return xent_loss

def vae_loss(x, x_decoded_mean):
    return kl_loss(x, x_decoded_mean) + logx_loss(x, x_decoded_mean)

vae = Model(inputs=x, outputs=_output)
optimizer = Adam(lr=learning_rate, decay=decay)
vae.compile(optimizer=optimizer, loss=vae_loss, metrics = [logx_loss, kl_loss])
vae.summary()

# load weights
vae.load_weights('Saved Tests/AE.synthetic.weights.h5')

# start = time()
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# save_weigths = ModelCheckpoint('AE.synthetic.weights.h5', verbose=1, monitor='val_loss',
#                                save_best_only=True, save_weights_only=False, mode='auto')
# csvlog = CSVLogger('AE.synthetic.log.csv', separator=',', append=True)
#
# # stop = EarlyStopping(monitor='kl_loss', min_delta=0, patience=0, verbose=1, mode='min', baseline=1.,
# #                      restore_best_weights=False)
# history = vae.fit(X_train, X_train,
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   verbose=1,
#                   validation_data= [x_test, x_test],
#                   callbacks=[tensorboard, save_weigths, csvlog])
#
# done = time()
# elapsed = done - start
# print("Elapsed: ", elapsed)

# vae.save('vae')

encoder = Model(x, z_mean)

g_z = Input(shape=(latent_dim,))
g_output = inst_layers(decoder_layers, g_z)
generator = Model(g_z, g_output)


encoder.save('encoder')
generator.save('decoder')
