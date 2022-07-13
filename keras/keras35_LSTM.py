from lightgbm import early_stopping
import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
#1.데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
# y = ? 

x = np.array(([1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9])) #(n,3) ->(n,3,1)로 바꿔서 연산한다. 데이터 양은 같다.잘라서 하는 연산때문.(3차원) 
y = np.array([4,5,6,7,8,9,10])

# RNN 에서                    
# x의_shape = (행,열, 몇개씩 자르는지!!!)  ,inputs:[batch, timesteps, feature] 
#output shape: unit 이라 지칭함

print(x.shape,y.shape)
x = x.reshape(7, 3, 1) # batch, timesteps, feature
print(x.shape)  
# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=66)

#2.모델구성
model = Sequential()
model.add(LSTM(200,activation='relu', input_shape=(3,1))) #input_shape 행무시 #dense 로 넘어갈 때 2차원으로 던져줌. 바로 dense로 받는거 가능(flatten x)
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(100, activation= 'relu'))
model.add(Dense(1))


model.summary()

#model.add(LSTM(10,activation='relu', input_shape=(3,1)))

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0
# ______________________________________________

# 
# Param: 541   input_dim 3
# simplernn * 4 = LSTM
# 숫자 4의 의미는 cell state,input gate,output gate,forget gate


#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
# earlyStopping = EarlyStopping(monitor= 'val_loss',patience=60, mode='min',restore_best_weights=True,verbose=1)
model.fit(x,y, epochs=500, batch_size=32,verbose=1)

#4.평가,예측 
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1)  # [[[8],[9],[10]]]  
result = model.predict(y_pred)  #모델은 3차원을 원한다.
print('loss : ', loss)
print('result : ', result)
