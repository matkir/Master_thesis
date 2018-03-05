from keras import layers
from keras import models
import numpy as np
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist


class fourmodels(object):
    def __init__(self, img_rows=720, img_cols=576, channel=3):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        
        self.input_shape = (self.img_rows, self.img_cols, self.channel)        
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        optimizer = Adam(0.0002, 0.5)
        
        
        self.discriminator = self.make_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                       optimizer=optimizer,
                                       metrics=['accuracy'])
    
        # Build and compile the generator
        self.generator = self.make_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        # The generator takes noise as input and generated imgs        
        z = Input(shape=self.input_shape)
        img = self.generator(z)       
    
        # For the combined model we will only train the generator
        self.discriminator.trainable = False        
        
        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)
    
    
        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.train(10)
        
    def make_discriminator(self):
        if self.D:
            #if exists, dont make new
            return self.D
        self.D = Sequential() #start sequential
        depth = 64
        dropout = 0.4
        # In: 720 x 576 x 3, depth = 1
        # Out: 718 x 574 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=4, input_shape=input_shape,\
                          padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=3, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        
        img = Input(shape=(self.img_rows, self.img_cols, self.channel))
        validity = self.D(img)
    
        return Model(img, validity)        
        

    def make_generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = (64+64+64+64)
        dim1 = 45
        dim2 = 36
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim1*dim2*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim1, dim2, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(3, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        
    
        img = Input(shape=(self.img_rows, self.img_cols, self.channel))
        validity = self.D(img)

        return Model(img, validity)        
        
    
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        
        self.img_height = 576
        self.img_width = 720
        self.channel = 3
        self.batch_size=512
        self.train_data_dir='../../../kvasir-dataset-v2/polyps'

        train_datagen = ImageDataGenerator() # remember that i can make img fancy
        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode=None)

        a=self.train_generator














"""
class img_class(object):
    def __init__(self):
        self.img_height = 576
        self.img_width = 720
        self.channel = 3
        self.batch_size=512
        self.train_data_dir='../../../kvasir-dataset-v2/polyps'

        train_datagen = ImageDataGenerator() # remember that i can make img fancy
        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode=None)
        
        
        
     
  
    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.train_generator.next()#[np.random.randint(0,self.train_generator[0].shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise, verbose=1)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                                     noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
"""
if __name__ == '__main__':
    mnist_dcgan = fourmodels()
    #mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    #mnist_dcgan.plot_images(fake=True)
    #mnist_dcgan.plot_images(fake=False, save2file=True)