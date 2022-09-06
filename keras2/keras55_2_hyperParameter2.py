# 하이퍼 파라미터 노드 추가, learning rate 추가 
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
import keras
import tensorflow as tf 

# 1.데이터
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
from keras.optimizers.optimizer_v2.adam import Adam

# 2.모델 
def build_model(drop=0.5, optimizer=Adam, activation='relu', node=111, lr=1):
    inputs = Input(shape= (28*28), name= 'input')
    x = Dense(node, activation=activation, name= 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    
    model.compile(optimizer=Adam, metrics=['acc'], loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam_v2', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu','linear','sigmoid','selu','elu']
    node = [32, 64, 128 ]
    lr = [0.01, 0.001, 0.0001]
    return {'batch_size': batchs, 'optimizer' : optimizers, 
            'drop' : dropout, 'activation': activation,
            'node': node, 'lr': lr}

hyperparameters = create_hyperparameter()
# print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
#케라스(텐서플로)모델을 사이킷런 모델로 래핑해준다.

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=3, verbose=1)

import time
start = time.time()
model.fit(x_train,y_train, epochs=5, validation_split=0.4)
end = time.time()

print('걸린시간 : ' , end - start)
print("model.best_param_: " , model.best_params_)    
print("model.best_estimator_: " , model.best_estimator_)    
print("model.best_score_: " , model.best_score_)    
print('model.score : ', model.score)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_test,y_predict))





    