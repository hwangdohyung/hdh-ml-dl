import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = "D:\study_data\_data/"
datasets = pd.read_csv(path + 'winequality-white.csv', index_col = None, header = 0, sep=';') # 분리

x = datasets.drop('quality', axis=1)
y = datasets['quality']


# 아웃라이어를 찾아서 NaN으로 바꾸기
def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

datasets = remove_outlier(datasets)

# # 이상치를 interpolate로 채우기
datasets = datasets.interpolate()


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


