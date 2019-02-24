from __future__ import print_function, division

import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import metrics
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sc
from load_barcodes_data import load_muenster, load_synthetic
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt

config = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.Session(config=config)

batch_size = 32


class BGAN():
    """Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/"""
    def __init__(self):
        self.img_rows = 180
        self.img_cols = 180
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.barcode_shape = (13, 10, 1)

        optimizer = Adam(0.00002, 0.0)

        # Build and compile the barcode discriminator
        self.discriminator = self.build_discriminator()
        # self.discriminator.load_weights('BGAN_B.Discriminator.weights.h5', by_name=True)

        self.discriminator.compile(loss=self.log_loss, optimizer=optimizer, metrics=[self.accuracy_bars])

        # Build the generator
        self.generator = self.build_generator()
        # self.generator.load_weights('BGAN_B.Generator.weights.h5', by_name=True)
        # The generator takes barcode as input and generated barcode IMAGE
        z = Input(shape=(self.barcode_shape))
        bar_img = self.generator(z)


        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated barcode image as input and determines validity in reference for the true barcode
        valid = self.discriminator(bar_img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # Means, generating images that D thinks they represent the true barcode
        self.combined = Model(z, valid)
        # self.combined.load_weights('BGAN_B.Combined.weights.h5', by_name=True)
        self.combined.compile(loss=self.boundary_loss, optimizer=optimizer, metrics=[self.accuracy_bars])

    def build_generator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.barcode_shape))
        model.add(Dense(1024, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.25))

        model.add(Dense(256 * 12 * 12, activation="relu"))
        model.add(Reshape((12, 12, 256)))

        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(UpSampling2D()) # (24,24)

        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(UpSampling2D()) # (48,48)

        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(UpSampling2D()) # (96,96)

        model.add(Conv2D(32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())  # (192,192)

        model.add(Conv2D(self.channels, kernel_size=4, padding="same")) # padding result (180,180)
        model.add(Activation("sigmoid"))
        model.add(Cropping2D((6,6)))

        model.summary()

        # model = Sequential()

        # model.add(Flatten(input_shape=self.barcode_shape))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.img_shape), activation='sigmoid')) # because the images are normalized [0,1]
        # model.add(Reshape(self.img_shape))
        #
        # model.summary()

        bar = Input(shape=self.barcode_shape)
        img = model(bar)

        return Model(bar, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, padding="same", strides=2, input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=4, padding="same", strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=4, padding="same", strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=4, padding="same", strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Dense(130, activation="relu"))
        model.add(Reshape(self.barcode_shape))
        # add softmax in the loss with different dim
        model.summary()

        bar_img_in = Input(shape=self.img_shape)
        disc_bar_out = model(bar_img_in)

        # the binary decision is added later in the training
        return Model(bar_img_in, disc_bar_out)

    def log_loss(self, y_true, y_pred):
        y_pred = K.softmax(y_pred[:,:,:,0], axis=-1)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = K.sum(y_true[:,:,:,0] * -K.log(y_pred), axis=-1, keepdims=False)
        # p_pred = K.sum(K.min(y_pred, axis=2), axis=1)  # probability that each barcode is correct
        # y_true = K.softmax(y_true, axis=2)
        # p_true = K.sum(K.min(y_true, axis=2), axis=1)  # probability that each barcode is correct
        # loss = K.categorical_crossentropy(y_true, y_pred)
        return K.mean(loss)

    def boundary_loss(self, y_true, y_pred):
        y_pred = K.softmax(y_pred[:,:,:,0], axis=-1)
        prob = K.prod(K.max(y_pred, axis=2), axis=1)  # probability that each barcode is correct
        prob = K.clip(prob, 1e-7, 1 - 1e-7)
        return 0.5 * K.mean((K.log(prob) - K.log(1 - prob))**2)

    def accuracy_bars(self, y_true, y_pred):
        # transfer discriminator output to one hot barcode
        y_pred = K.one_hot(K.argmax(K.softmax(y_pred[:, :, :, 0])), 10)
        acc = K.equal(y_true[:, :, :, 0], y_pred)
        acc = K.all(K.all(acc, axis=-1), axis=-1)
        return K.mean(acc)

    def named_logs(self, metrics_names, logs):
        result = {}
        for l in zip(metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def plot_bars(self, t_bars, p_bars, bar_imgs, epoch, name):
        # Plot a random sample of 9 test images, their predicted labels and ground truth
        fig = plt.figure(figsize=(15, 15))
        for i, index in enumerate(np.random.choice(bar_imgs.shape[0], size=9, replace=False)):
            ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
            # Display each image
            ax.imshow(np.squeeze(bar_imgs[index,:,:,:]), cmap='gray')
            predict_label = np.array2string(np.argmax(p_bars[index,:,:,0], axis=1), separator='')
            true_label = np.array2string(np.argmax(t_bars[index,:,:,0], axis=1), separator='')
            # Set the title for each image
            ax.set_title("{} ({})".format(predict_label[1:14],
                                          true_label[1:14]),
                         color=("green" if predict_label[1:14] == true_label[1:14] else "red"))
        fig.savefig("GAN Images/%s epoch_%d.png" % (name, epoch))
        plt.close()

    def train(self, epochs, batch_size=128):

        # Load the dataset
        X_train, y_train = load_synthetic()

        # Callbacks
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()),
                                  histogram_freq=0,
                                  batch_size=batch_size,
                                  write_graph=True,
                                  write_grads=True
                                  )
        save_G_weigths = ModelCheckpoint('BGAN_B.Generator.weights.h5', verbose=0, monitor='G_loss',
                                       save_best_only=True, save_weights_only=False, mode='min')
        save_D_weigths = ModelCheckpoint('BGAN_B.Discriminator.weights.h5', verbose=0, monitor='D_loss',
                                         save_best_only=True, save_weights_only=False, mode='min')
        save_C_weigths = ModelCheckpoint('BGAN_B.Combined.weights.h5', verbose=0, monitor='G_loss',
                                         save_best_only=True, save_weights_only=False, mode='min')

        tensorboard.set_model(self.combined)  # either one of the models is fine becuase they finish at the same epoch
        save_G_weigths.set_model(self.generator)
        save_D_weigths.set_model(self.discriminator)
        save_C_weigths.set_model(self.combined)

        false_bars = to_categorical(np.zeros((batch_size, 13)), num_classes=10)
        false_bars = np.expand_dims(false_bars, -1)

        for epoch in range(epochs):

            # Adversarial ground truths
            idx = np.random.randint(0, y_train.shape[0], batch_size)
            bars = y_train[idx, :]

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a the batch of barcode images which matches the valid
            true_bar_imgs = X_train[idx, :, :, :]

            # Generate a batch of new barcode
            gen_bar_imgs = self.generator.predict(bars)

            # Train the discriminator - where is
            for _ in range(10):  # Train D 10 times more than G
                d_loss_real_true = self.discriminator.train_on_batch(true_bar_imgs, bars) # hope loss 0 acc 100% - D recognize the barcode when the image is real
            d_loss_gen_true = self.discriminator.train_on_batch(gen_bar_imgs, false_bars) # (*) hope loss 0 acc 100% - D recognize that this is not a barcode when the image is fake
            d_loss = 0.5 * np.add(d_loss_real_true, d_loss_gen_true)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(1):  # Train D 10 times more than G
                g_loss = self.combined.train_on_batch(bars, bars) # This forms an adversary to (*) - hope loss 0 acc 100%
            # Means G creates images that D thinks they represent the true barcode

            # Plot the progress
            print("%d D loss: [Real: %f, Gen: %f] D acc.: [Real: %.2f%% Gen: %.2f%%] [G loss: %f G acc: %.2f%%]" %
                  (epoch, d_loss_real_true[0], d_loss_gen_true[0], 100 * d_loss_real_true[1], 100 * d_loss_gen_true[1],
                   g_loss[0], 100 * g_loss[1]))

            losses = np.asarray([d_loss_real_true[0], d_loss_gen_true[0], 100 * d_loss_real_true[1],
                                 100 * d_loss_gen_true[1], g_loss[0], 100 * g_loss[1]])
            names = ['D loss real', 'D loss gen', 'D acc real', 'D acc gen', 'G loss', 'G acc']

            tensorboard.on_epoch_end(epoch, self.named_logs(names, losses))
            # save_G_weigths.on_epoch_end(epoch, {'G_loss': g_loss[0]})
            # save_D_weigths.on_epoch_end(epoch, {'D_loss': d_loss[0]})
            # save_C_weigths.on_epoch_end(epoch, {'G_loss': g_loss[0]})

            if epoch % 250 == 0:
                p_dis_true = self.discriminator.predict(true_bar_imgs)
                self.plot_bars(bars, p_dis_true, true_bar_imgs, epoch, 'Real')

                p_dis_gen = self.discriminator.predict(gen_bar_imgs)
                self.plot_bars(false_bars, p_dis_gen, gen_bar_imgs, epoch, 'Gen')

                gen_bar_imgs = self.generator.predict(bars)
                p_com = self.combined.predict(bars)
                self.plot_bars(bars, p_com, gen_bar_imgs, epoch, 'Comb')

        tensorboard.on_train_end(None)
        save_G_weigths.on_train_end(None)
        save_D_weigths.on_train_end(None)

if __name__ == '__main__':
    bgan = BGAN()
    bgan.train(epochs=10000, batch_size=batch_size)