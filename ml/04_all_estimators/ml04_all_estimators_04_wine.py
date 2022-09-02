from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_wine()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)#스케일링한것을 보여준다.
x_test = scaler.transform(x_test)#test는 transfrom만 해야됨 

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
        model.fit(x_train,y_train)
    
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test,y_predict)
        print(name, '의 정답율 : ', acc)              
    except:                                   # 에러가 뜨면 계속 진행해
        continue
        # print(name, '안나온놈')

# AdaBoostClassifier 의 정답율 :  0.8703703703703703
# BaggingClassifier 의 정답율 :  0.9814814814814815
# BernoulliNB 의 정답율 :  0.8703703703703703
# CalibratedClassifierCV 의 정답율 :  0.9629629629629629
# DecisionTreeClassifier 의 정답율 :  0.8148148148148148
# DummyClassifier 의 정답율 :  0.3888888888888889
# ExtraTreeClassifier 의 정답율 :  0.7777777777777778
# ExtraTreesClassifier 의 정답율 :  1.0
# GaussianNB 의 정답율 :  0.9629629629629629
# GaussianProcessClassifier 의 정답율 :  0.9629629629629629
# GradientBoostingClassifier 의 정답율 :  0.9074074074074074
# HistGradientBoostingClassifier 의 정답율 :  1.0
# KNeighborsClassifier 의 정답율 :  0.8703703703703703
# LabelPropagation 의 정답율 :  0.9074074074074074
# LabelSpreading 의 정답율 :  0.9074074074074074
# LinearDiscriminantAnalysis 의 정답율 :  0.9814814814814815
# LinearSVC 의 정답율 :  0.9629629629629629
# LogisticRegression 의 정답율 :  0.9814814814814815
# LogisticRegressionCV 의 정답율 :  0.9629629629629629
# MLPClassifier 의 정답율 :  0.9814814814814815
# NearestCentroid 의 정답율 :  0.9259259259259259
# NuSVC 의 정답율 :  0.9629629629629629
# PassiveAggressiveClassifier 의 정답율 :  0.9629629629629629
# Perceptron 의 정답율 :  0.9444444444444444
# QuadraticDiscriminantAnalysis 의 정답율 :  0.9629629629629629
# RandomForestClassifier 의 정답율 :  1.0
# RidgeClassifier 의 정답율 :  0.9814814814814815
# RidgeClassifierCV 의 정답율 :  0.9814814814814815
# SGDClassifier 의 정답율 :  0.9629629629629629
# SVC 의 정답율 :  0.9629629629629629