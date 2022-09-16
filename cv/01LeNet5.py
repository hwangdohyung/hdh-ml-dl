from keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,MaxPooling2D,Conv2D,Input,Flatten

input1 = Input(shape=(28,28,1))
l1 = Conv2D(6,kernel_size=(5,5),padding='same')(input1)
l2 = MaxPooling2D()(l1)
l3 = Conv2D(16,4)(l2)
l4 = MaxPooling2D()(l3) 
l5 = Flatten()(l4)
l6 = Dense(120)(l5)
l7 = Dense(80)(l6)
l8 = Dense(10,activation='softmax')(l7)
model = Model(input1,l8)

model.compile(loss= 'sparse_categorical_crossentropy',optimizer= 'adam')
model.fit(x_train,y_train,epochs=2,batch_size=32)
import numpy as np 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)

print(y_predict,y_test)
print(y_predict.shape,y_test.shape)
from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(y_predict,y_test))

# acc :  0.9541


