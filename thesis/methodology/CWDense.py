from keras.engine.topology import Layer
import keras.backend as K

class CWDense(Layer):
    def __init__(self, **kwargs):
        super(CWDense, self).__init__(**kwargs)
    
    def build(self,input_shape):
        _, self.width, self.height, self.n_feat_map = input_shape
        self.kernel = self.add_weight("CWDense",
                                        shape=(self.n_feat_map,
                                        self.width*self.height,
                                        self.width*self.height),
                                        initializer='glorot_uniform',
                                        trainable=True)        
        super(CWDense, self).build(input_shape)

    def call(self, x):
        x = tf.reshape(x,[-1,self.width*self.height,self.n_feat_map])
        x = tf.transpose( x, [2,0,1] )

        x = K.dot(x,self.kernel)

        x = tf.transpose(x, [1,2,0])
        x = tf.reshape(x,[-1,self.height,self.width,self.n_feat_map])
        return x