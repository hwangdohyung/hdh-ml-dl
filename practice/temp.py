from tensorflow.python.keras.layers import Dense,Activation,Flatten,Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Concatenate


class Mymodel():
    def __init__(self, units =30, activation = 'relu', **kwargs):
        super(Mymodel, self).__init__(**kwargs)
        self.dense_layer1 = Dense(300,activation=activation)
        self.dense_layer2 = Dense(100,activation=activation)
        self.dense_layer3 = Dense(30,activation=activation)
        self.output_layer= Dense(10,activation='softmax')
        

    def call(self, inputs):
        x = self.dense_layer1(inputs)
        x = self.dense_layer2(x)
        x = self.dense_layer3(x)
        x = self.output_layer(x)
        return x        





