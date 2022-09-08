from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

#1.
# model.trainable = False

#2.
# for layer in model.layers:
#     layer.trainable = False
   
# model.layers[0].trainable = False    #dense 
# model.layers[1].trainable = False    #dense_1 
model.layers[2].trainable = False    #dense_2 
     
model.summary()

print(model.layers)




