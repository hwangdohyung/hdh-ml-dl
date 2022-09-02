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
print('model.score: ', score)                           # model.score:  0.7326530612244898
print('acc_score : ', accuracy_score(y_test,y_predict)) # acc_score :  0.7255102040816327

print('f1_score(macro) : ', f1_score(y_test,y_predict,average='macro')) # f1_score는 원래 2진분류용, 다중분류로 쓰기위해 average를 포함해줌  0.4452474171560578
print('f1_score(micro) : ', f1_score(y_test,y_predict,average='micro')) # 0.7306122448979592 ,acc와 똑같은값

#과제 f1 score에 대한 이해 






















