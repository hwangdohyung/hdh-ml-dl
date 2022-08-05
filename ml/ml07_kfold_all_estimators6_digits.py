from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_digits()

x = datasets['data']
y = datasets['target']


# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle= True, random_state=66)



from sklearn.utils import all_estimators

#2.모델구성
allAlgorithms = all_estimators(type_filter='classifier')  # 분류모델 전부를 보여준다    
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms))
import warnings
warnings.filterwarnings('ignore') # 출력만 안해준다

for (name, algorithm) in allAlgorithms:       # 리스트 안에 키밸류(알고리즘 이름과,위치)를 받아서 반복한다.
    try:                                      # 이것을 진행해 
        model = algorithm()
  
        acc = cross_val_score(model, x,y , cv=kfold)
        print(name, '의 정답율 : ', acc)   
                   
    except:                                   # 에러가 뜨면 계속 진행해
        continue
        # print(name, '안나온놈')

