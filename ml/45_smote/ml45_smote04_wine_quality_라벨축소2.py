import pandas as pd 
import numpy as np 

# 1.데이터
datasets = pd.read_csv('D:/study_data/_data/winequality-white.csv',index_col=None,header=0,sep=';')

print(datasets.shape)                   #(4898, 12)
print(datasets.describe())
print(datasets.info())

datasets2 = datasets.values                 # 첫번째 방법 .values
# datasets = datasets.to_numpy()            # 두번째 방법 .to_numpy()
print(type(datasets2))                      # type 확인 (numpy로 바뀌면 index,컬럼명 삭제)

x = datasets2[:, :11]                       # 모든 행, 10번재 열까지
y = datasets2[:, 11]                        # 모든 행, 11번째 열
print(x.shape,y.shape)                      #(4898, 11) (4898,)

print(np.unique(y,return_counts=True))      #(array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(datasets['quality'].value_counts())   #pandas 는 value_counts

# 업무할때는 오너와의 소통을 통해 라벨 갯수를 축소 시키는것을 생각해볼 수 있음 데이터가 너무 적은 라벨이 있기 때문
print(y[:20])


for index, value in enumerate(y):
    if value == 9 :
        y[index] = 7
    elif value == 8 :
        y[index] = 7
    elif value ==  7:
        y[index] = 7
    elif value == 6 :
        y[index] = 6
    elif value == 5 :
        y[index] = 5
    elif value == 4 :
        y[index] = 4
    elif value == 3 :
        y[index] = 4
    else : 
        y[index] = 0

print('======================')
print(np.unique(y, return_counts=True))



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True,stratify=y)

# 2.모델구성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# 3.훈련 
model.fit(x_train,y_train)

# 4.평가, 예측 
from sklearn.metrics import accuracy_score,f1_score
y_predict = model.predict(x_test)
score = model.score(x_test,y_test)
# print('model.score: ', score)                           # model.score:  0.7326530612244898
print('acc_score : ', accuracy_score(y_test,y_predict)) # acc_score :  0.7255102040816327
print('f1_score(macro) : ', f1_score(y_test,y_predict,average='macro')) # f1_score(macro) :  0.7595647014889432
# print('f1_score(micro) : ', f1_score(y_test,y_predict,average='micro')) # 0.7306122448979592 ,acc와 똑같은값

#######################################################
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
print('=========== SMOTE 적용후 ==============')
smote = SMOTE(random_state=123,k_neighbors=20) #k_neighbors 디폴트 :5
x_train,y_train = smote.fit_resample(x_train,y_train)
# 가장 큰 숫자에 통일되서 증폭됨 ,단점 (resampling과정)은 데이터가 많아져서 오래걸림(전체를 /n 해서 10번반복하여서 하는 편법으로 시간을 줄일수 있다.)
print(pd.Series(y_train).value_counts())   

model= RandomForestClassifier()
model.fit(x_train, y_train)     #당연히 평가데이터는 증폭시킬 필요 x

y_predict = model.predict(x_test)
score = model.score(x_test,y_test)
# print('model.score: ', score)                           
print('acc_score : ', accuracy_score(y_test,y_predict)) 
print('f1_score(macro) : ', f1_score(y_test,y_predict,average='macro')) 







