from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from scipy import stats
import matplotlib.pyplot as plt

import sys, os
import cv2
from tqdm import tqdm

import numpy as np

class DCGAN():
    def __init__(self):
        self.img_rows = 240#720 
        self.img_cols = 192#576
        self.channels = 3

        optimizer = Adam(0.002, 0.5)




        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.discriminator.compile(loss='binary_crossentropy', 
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)


        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(60*48*3, activation="relu", input_shape=noise_shape))
        model.add(Reshape((60, 48, 3)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8)) 
        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=2, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(3, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        #(X_train, _), (_, _) = mnist.load_data()
    
        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)
        
        X_train=load_polyp_data()
        
        half_batch=int(len(X_train)/200)


        for epoch in tqdm(range(epochs)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch,100))
            #noise = np.random.normal(0, 1, (1, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = stats.threshold(gen_imgs, threshmax=1, newval=1)
        gen_imgs = stats.threshold(gen_imgs, threshmin=1, newval=0)
        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        try: 
            for i in range(r):
                for j in range(c):
                    if i==0 and j==0 and False:
                        axs[i,j].imshow(X_train[cnt, :,:,:])
                        axs[i,j].axis('off')
                        cnt += 1
                        print("asff")
                    else:    
                        axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                        axs[i,j].axis('off')
                        cnt += 1
            fig.savefig("dcgan/images/mnist_%d.png" % epoch)
            plt.close()
        except:
            print("WRONG")
 



def load_polyp_data():
    if False: 
        return np.load("train_data.npy")
    data=np.ndarray(shape=(1000, int(240), int(192), 3),dtype=np.int32)
    folder ='../../../kvasir-dataset-v2/polyps' 
    i=0
    for img in tqdm(os.listdir(folder)):
        path=os.path.join(folder,img)
        save=cv2.imread(path)
        save=cv2.resize(save,(int(192),int(240)))
        data[i]=(np.array(save))
        i+=1
    np.save("train_data.npy", data)
    return data
if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=100, batch_size=32, save_interval=5)
   