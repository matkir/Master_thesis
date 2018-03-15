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
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class AE():
     def __init__(self):
          self.img_rows = 720//2 # Original is ~720 
          self.img_cols = 576//2 # Original is ~576
          self.channels = 3   # RGB 
          self.img_shape=(self.img_rows,self.img_cols,self.channels)
          self.buld_AE()
          if '-w' in sys.argv: 
               self.decoder.load_weights("decoder_weights.h5")
               self.encoder.load_weights("encoder_weights.h5")
               self.autoencoder.load_weights("ae_weights.h5")               
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
               Encoder=Model(self.input_img, self.encoded,name='encoder')
               return Encoder
          def build_decoder():
               self.input_code=Input(shape=(15,12,1))
               x = Conv2D(1, (3, 3), activation='relu', padding='same')(self.input_code)
               x = UpSampling2D((3, 3))(x)
               x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
               x = UpSampling2D((2, 2))(x)
               x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
               x = UpSampling2D((2, 2))(x)
               x = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
               x = UpSampling2D((2, 2))(x)
               self.decoded = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
               Decoder=Model(self.input_code, self.decoded,name='decoder')
               return Decoder
          def build_AE(e,d):
               AE=Sequential()
               AE.add(e)
               AE.add(d)
               AE.compile(optimizer='adam', loss='mse')
               return AE
          self.encoder=build_encoder()
          self.decoder=build_decoder()
          self.autoencoder=build_AE(self.encoder, self.decoder) 
          
          #end of decoder
          
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
               idx2 = np.random.randint(0, X_train.shape[0], batch_size)
               imgs = X_train[idx] 
               imgs2 = X_train[idx] 
               loss=self.autoencoder.train_on_batch(imgs, imgs)
               loss2=self.autoencoder.test_on_batch(imgs2, imgs2)
               print(loss,loss2)
               if epoch % save_interval == 0:
                    self.save_imgs(epoch,imgs[0:3,:,:,:])               
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
               
          #noise_enc=np.clip((np.random.normal(img,0.01)),-1,1)
          gen_enc = self.encoder.predict(img)
          
          noise_dec = np.random.normal(0, 1, (3,self.img_shape[0]//24,self.img_shape[1]//24,1))
          gen_dec = self.decoder.predict(noise_dec)
          
          gen_ae=self.autoencoder.predict(img)

          self.plot_1_to_255(gen_enc, gen_dec, gen_ae,epoch)     
     
     
     
     def load_polyp_data(self):
          if '-l' in sys.argv:
               return np.load("train_data.npy")
          data=np.ndarray(shape=(2000, self.img_shape[0], self.img_shape[1], self.img_shape[2]),dtype=np.int32)
          folder ='../../../kvasir-dataset-v2/blanding' 
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
     
     def plot_1_to_255(self,enc_img,dec_img,ae_img,epoch):
               fig, axs = plt.subplots(3, 3) #3 of each picture
               #dec_img=(dec_img*127.5)+127.5
               #enc_img=np.squeeze((enc_img*127.5)+127.5) #remove silly dim
               #ae_img=(ae_img*127.5)+127.5
               dec_img=(dec_img*0.5)+0.5
               enc_img=np.squeeze((enc_img*0.5)+0.5) #remove silly dim
               ae_img=(ae_img*0.5)+0.5
               cnt1=0
               cnt2=0
               cnt3=0
               for i in range(3):
                    for j in range(3):
                         if j==0:
                              axs[i,j].imshow(dec_img[cnt1, :,:,:])
                              axs[i,j].axis('off')
                              cnt1 += 1
                         elif j==1:
                              axs[i,j].imshow(enc_img[cnt2, :,:])
                              axs[i,j].axis('off')
                              cnt2 += 1
                         elif j==2:
                              axs[i,j].imshow(ae_img[cnt3, :,:,:])
                              axs[i,j].axis('off')
                              cnt3 += 1
                         else:
                              raise IndexError    
               
               plt.suptitle('decoded img | encoded img | encoded then decoded', fontsize=16)
               fig.savefig("images/mnist_%d.png" % epoch)
               plt.close()               
         

if __name__ == '__main__':
     if '-i' in sys.argv:
          data=np.load("train_data.npy")
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
     obj.train(epochs=a, batch_size=80, save_interval=5)















