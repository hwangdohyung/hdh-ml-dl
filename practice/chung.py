import pandas as pd
import numpy as np
import glob
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,GRU,Dense,Dropout
from sklearn.model_selection import train_test_split

path = 'D:\study_data\_data\dacon_chung/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))


test_input_list2 = sorted(glob.glob(path + 'test_input/*.csv'))
test_target_list2 = sorted(glob.glob(path + 'test_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  

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

train_data, label_data = aaa(train_input_list, train_target_list) #, False)
val_data, val_target = aaa(val_input_list, val_target_list) #, False)
test_data,test_target = aaa(test_input_list2, test_target_list2) #, False)


print(len(train_data), len(label_data)) # 1607 1607
print(len(train_data[0]))   # 1440
print(label_data)   # 1440
print(train_data.shape, label_data.shape)   # (1607, 1440, 37) (1607,)
print(val_data.shape, val_target.shape)   # (206, 1440, 37) (206,)


train_data = np.save('D:\study_data\_save\_npy\green/train_data1.npy',arr=train_data)
label_data = np.save('D:\study_data\_save\_npy\green/label_data2.npy',arr=label_data)
val_data = np.save('D:\study_data\_save\_npy\green/val_data1.npy',arr=val_data)
val_target = np.save('D:\study_data\_save\_npy\green/val_target1.npy',arr=val_target)
test_data = np.save('D:\study_data\_save\_npy\green/test_data1.npy',arr=test_data)
test_target = np.save('D:\study_data\_save\_npy\green/test_target1.npy',arr=test_target)


x_train,x_test,y_train,y_test = train_test_split(train_data,label_data,train_size=0.91,shuffle=False)


#2. 모델 구성
model = Sequential()
model.add(GRU(100,input_shape=(1440,37)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()

import time
start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=3, batch_size=700, 
                validation_data=(val_data, val_target),
                verbose=2,#callbacks = [earlyStopping]
                )
end_time = time.time()-start_time
#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


# y_predict = model.predict(x_test)
# print(y_test.shape) #(152,)
# print(y_predict.shape) #(152, 13, 1)

# from sklearn.metrics import accuracy_score, r2_score,accuracy_score
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)


print(test_input_list2)
model.fit(train_data,label_data)
y_summit = model.predict(test_data)


path2 = 'D:\study_data\_data\dacon_chung\\test_target/' # ".은 현재 폴더"
targetlist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv','TEST_06.csv']
# [29, 35, 26, 32, 37, 36]
empty_list = []
for i in targetlist:
    test_target2 = pd.read_csv(path2+i)
    empty_list.append(test_target2)
    
empty_list[0]['rate'] = y_summit[:29]
empty_list[0].to_csv(path2+'TEST_01.csv')
empty_list[1]['rate'] = y_summit[29:29+35]
empty_list[1].to_csv(path2+'TEST_02.csv')
empty_list[2]['rate'] = y_summit[29+35:29+35+26]
empty_list[2].to_csv(path2+'TEST_03.csv')
empty_list[3]['rate'] = y_summit[29+35+26:29+35+26+32]
empty_list[3].to_csv(path2+'TEST_04.csv')
empty_list[4]['rate'] = y_summit[29+35+26+32:29+35+26+32+37]
empty_list[4].to_csv(path2+'TEST_05.csv')
empty_list[5]['rate'] = y_summit[29+35+26+32+37:29+35+26+32+37+36]
empty_list[5].to_csv(path2+'TEST_06.csv')


import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:\study_data\_data\dacon_chung\\test_target")
with zipfile.ZipFile("D:\study_data\_data\dacon_chung/submission32.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()
print('Done')
print('걸린 시간:', end_time)


print(x.shape )


