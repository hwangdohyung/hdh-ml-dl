from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) #(569, 30)
# print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

# # minmax , standard
# # scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

#2.모델구성
input1 = Input(shape=(30,))
dense1 = Dense(50,activation= 'relu')(input1)
dense2 = Dense(50,activation= 'relu')(dense1)
dense3 = Dense(50,activation= 'relu')(dense2)
dense4 = Dense(40,activation= 'relu')(dense3)
dense5 = Dense(30,activation= 'relu')(dense4)
output1 = Dense(1,activation='sigmoid')(dense5)
model= Model(inputs=input1,outputs=output1)


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping= EarlyStopping(monitor= 'val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1) 

model.load_weights('./_save/keras23_11_save_cancer.h5')

#3.컴파일,훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mse'],)

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss)

y_predict = model.predict(x_test) # 반올림을 해줘야 acc가 나온다 
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict) 
print('acc스코어: ', acc)

