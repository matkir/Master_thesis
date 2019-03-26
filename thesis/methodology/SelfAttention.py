class Attention(Layer): 
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(Attention, self).__init__(**kwargs)

    def build(self,filters):
        self.kernelf = self.add_weight(name='kernelf', 
                            shape=(1,1,self.filters,self.filters),
                            initializer='uniform',trainable=True)
        self.kernelg = self.add_weight(name='kernelg', 
                            shape=(1,1,self.filters,self.filters),
                            initializer='uniform',trainable=True)
        self.kernelh = self.add_weight(name='kernelh', 
                            shape=(1,1,self.filters,self.filters),
                            initializer='uniform',trainable=True)
        super(Attention, self).build(filters)

    def call(self, x):
        f = K.conv2d(x, self.kernelf, strides=1, padding="same")
        g = K.conv2d(x, self.kernelg, strides=1, padding="same")
        h = K.conv2d(x, self.kernelh, strides=1, padding="same")
        
        f = tf.transpose(f,perm=[0,1,2,3])
        i = f*g
        j = K.softmax(i)
        k = j*h
        
        return k