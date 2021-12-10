from __future__ import print_function, division
import matplotlib.pyplot as plt
import sys
import numpy as np
import scipy
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

class DCGAN():
  
    def __init__(self, dataset_path, img_rows, img_cols, channels, convolution_scale):
        
        # Input shape
        self.img_rows = img_rows
        self.dataset_path = dataset_path
        self.img_cols = img_cols
        self.channels = channels
        self.convolution_scale = convolution_scale
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_data(self):
        
        img_res = (self.img_rows, self.img_cols) 
        imgs = []
        for img_path in os.listdir(self.dataset_path):
            img = self.imread(os.path.join(self.dataset_path, img_path))
            img = scipy.misc.imresize(img, img_res)
            imgs.append(img)
            
        imgs = np.array(imgs)/127.5 - 1.
        
        return imgs
      
    def imread(self, path):
        if self.channels == 1:
            return scipy.misc.imread(path, mode='L').astype(np.float)
        
        else:
            return scipy.misc.imread(path, mode='RGB').astype(np.float)
    
    def build_generator(self):
        
        img_kernel = int(self.img_cols/4)
        k = int(self.convolution_scale)
        model = Sequential()
        model.add(Dense(128*k*img_kernel*img_kernel, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((img_kernel, img_kernel, 128* k)))
        model.add(UpSampling2D())
        model.add(Conv2D(128* k, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64*k, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img, name="generator")

    def build_discriminator(self):

        model = Sequential()
        k = int(self.convolution_scale)
        model.add(Conv2D(32*k, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64*k, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128*k, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256*k, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity, name="discriminator")

    def train(self, epochs, batch_size, save_path, save_interval=50):
        
        self.batch_size = batch_size
        self.save_path = save_path
        
        # Load the dataset
        X_train = self.load_data()
    
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        print(X_train.shape)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % 100 == 0:
                self.save_imgs(epoch)
                self.generator.save(os.path.join(self.save_path, "%d model.h5" % (epoch)))

    def save_imgs(self, epoch):
        
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.save_path,"icon_%d.png" % epoch))
        plt.close()
