import pandas as pd
import numpy as np
import glob

path = 'D:\study_data\_data\dacon_chung/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))
test_input_list = sorted(glob.glob(path + 'test_input/*.csv'))
test_target_list = sorted(glob.glob(path + 'test_target/*.csv'))


train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

x_train, y_train = aaa(train_input_list, train_target_list) #, False)
x_test, y_test = aaa(val_input_list,val_target_list)
lat_x,lat_y = aaa(test_input_list,test_target_list)


print(x_train[0])
print(len(x_train), len(y_train)) # 1607 1607
print(len(x_train[0]))   # 1440
print(y_train)   # 1440
print(x_train.shape, y_train.shape)   # (1607, 1440, 37) (1607,)
print(x_test.shape, y_test.shape)     # (206, 1440, 37) (206,)


from xgboost import XGBRegressor
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM,Conv1D,Flatten,Conv2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping


# x_train = x_train.reshape(1607,1440*37)
# x_test = x_test.reshape(206,1440*37)
# lat_x = lat_x.reshape(195,1440*37)


# model = XGBRegressor()
# model.fit(x_train,y_train)
# print("gg : ",  model.score(x_test,y_test))

# submit= model.predict(lat_x)

# print(submit.shape)

#2.모델구성
model = Sequential()
model.add(Conv1D(256,2,activation= 'relu',input_shape=(1440,37))) 
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(1))


#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor= 'loss',patience=40, mode='min',restore_best_weights=True,verbose=1)
model.fit(x_train,y_train, epochs=500, batch_size=32,verbose=1,callbacks=earlyStopping)

#4.평가,예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)


submit = model.predict(lat_x)

import os
import zipfile
os.chdir("D:\study_data\_data\dacon_chung/test_target/")
submission = zipfile.ZipFile("../submission.zip", 'w')
for path in test_target_list:
    path = path.split('/')[-1]
    submission.write(path)
submission.close()

