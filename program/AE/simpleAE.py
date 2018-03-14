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
#os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu'

class AE():
     def __init__(self):
          self.img_rows = 720//2 # Original is ~720 
          self.img_cols = 576//2 # Original is ~576
          self.channels = 3   # RGB 
          self.img_shape=(self.img_rows,self.img_cols,self.channels)
          self.buld_AE()
     
     def buld_AE(self):
          def build_encoder():
               self.input_img = Input(shape=(self.img_rows, self.img_cols, self.channels)) 
               x = Conv2D(16, (3, 3), activation='relu', padding='same')(self.input_img)
               x = MaxPooling2D((2, 2), padding='same')(x)
               x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
               x = MaxPooling2D((2, 2), padding='same')(x)
               x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
               x = MaxPooling2D((2, 2), padding='same')(x)
               x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
               self.encoded = MaxPooling2D((3, 3), padding='same')(x)
               Encoder=Model(self.input_img, self.encoded)
               return Encoder
          def build_decoder():
               self.input_code=Input(shape=(15,12,1))
               x = Conv2D(1, (3, 3), activation='relu', padding='same')(self.input_code)
               x = UpSampling2D((3, 3))(x)
               x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
               x = UpSampling2D((2, 2))(x)
               x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
               x = UpSampling2D((2, 2))(x)
               x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
               x = UpSampling2D((2, 2))(x)
               self.decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
               Decoder=Model(self.input_code, self.decoded)
               return Decoder
          def build_AE(e,d):
               AE=Sequential()
               AE.add(e)
               AE.add(d)
               AE.compile(optimizer='adadelta', loss='mse')     
               return AE
          self.encoder=build_encoder()
          self.decoder=build_decoder()
          self.autoencoder=build_AE(self.encoder, self.decoder) 
          
          #end of decoder
          
          """
          self.autoencoder = Model(self.input_img, self.decoded)
          self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
          self.autoencoder.summary()
          # this model maps an input to its encoded representation
          self.encoder = Model(self.input_img, self.encoded)
          
          # create a placeholder for an encoded (32-dimensional) input
          decoding_dim=(self.img_rows//24,self.img_cols//24,1,)
          decoded_input = Input(shape=(15,12,3,))
          
          # retrieve the last layer of the autoencoder model
          decoder_layer = self.autoencoder.layers[-1]
          # create the decoder model
          self.decoder = Model(decoded_input, decoder_layer(decoded_input))
          """
          if '-s' in sys.argv:          
               print("autoencoder")
               self.autoencoder.summary()
               print("decoder")
               self.decoder.summary()
               print("encoder")
               self.encoder.summary()
          
     def train(self, epochs=20, batch_size=32, save_interval=5):
          X_train=self.load_polyp_data()
          for epoch in tqdm(range(epochs)):
               idx = np.random.randint(0, X_train.shape[0], batch_size)
               imgs = X_train[idx] 
               loss=self.autoencoder.train_on_batch(imgs, imgs)
               print(loss)
               if epoch % save_interval == 0:
                    self.save_imgs(epoch,imgs[0:30,:,:,:])               
          # encode and decode some digits
          # note that we take them from the *test* set
          print("saving")
          self.decoder.save("new_decoder.h5")
          self.encoder.save("new_encoder.h5")
          self.autoencoder.save("new_ae.h5")
          self.decoder.save_weights("decoder_weights.h5")
          self.encoder.save_weights("encoder_weights.h5")
          self.autoencoder.save_weights("ae_weights.h5")
          
          
     def save_imgs(self, epoch,img):
          r, c = 5, 5
          #noise_enc = np.random.normal(0, 1, (r * c,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
          #gen_enc = self.encoder.predict(img)
          
          #noise_dec = np.random.normal(0, 1, (r * c,self.img_shape[0]//24,self.img_shape[1]//24,1))
          #gen_dec = self.decoder.predict(gen_enc)
          # Rescale images 0 - 1
          #gen_enc = 0.5 * gen_enc + 0.5
          #gen_dec = 0.5 * gen_dec + 0.5
          gen_dec=self.autoencoder.predict(img)*127.5 
          gen_dec=gen_dec+127.5
          fig, axs = plt.subplots(r, c)
          #fig.suptitle("DCGAN: Generated digits", fontsize=12)
          cnt = 0
          try: 
               for i in range(r):
                    for j in range(c):
                         #if cnt==2:
                         #     axs[i,j].imshow(np.squeeze(gen_enc[0, :,:,:]))
                         #     axs[i,j].axis('off')
                         #     cnt += 1
                         #else:
                         axs[i,j].imshow(gen_dec[cnt, :,:,:])
                         axs[i,j].axis('off')
                         cnt += 1
               fig.savefig("images/mnist_%d.png" % epoch)
               plt.close()
          except Exception as e: 
               print(e)
     
     
     
     
     def load_polyp_data(self):
          if '-l' in sys.argv:
               return np.load("train_data.npy")
          data=np.ndarray(shape=(1000, self.img_shape[0], self.img_shape[1], self.img_shape[2]),dtype=np.int32)
          folder ='../../../kvasir-dataset-v2/polyps' 
          i=0
          for img in tqdm(os.listdir(folder)):
               path=os.path.join(folder,img)
               save=cv2.imread(path)
               save=cv2.resize(save,(self.img_shape[1],self.img_shape[0]))
               data[i]=(np.roll(np.array(save),1,axis=-1))
               i+=1
          data = (data.astype(np.float32) - 127.5) / 127.5
          np.save("train_data.npy", data)
          return data
     

if __name__ == '__main__':
     if '-i' in sys.argv:
          data=np.load("train_data.npy")
          #a = (data.astype(np.float32) - 127.5) / 127.5
          data = 0.5 * data + 0.5
               
          plt.imshow(data[np.random.randint(999),:,:,:])
          plt.show()
          sys.exit()
        
     obj = AE()
     a=sys.argv[1]
     if int(a):
          a=int(a)
     else:
          a=50
     obj.train(epochs=a, batch_size=32, save_interval=5)















