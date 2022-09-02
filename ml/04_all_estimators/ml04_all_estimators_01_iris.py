from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_iris()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.7,random_state=31)

scaler = MinMaxScaler()
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

# AdaBoostClassifier 의 정답율 :  0.9333333333333333
# BaggingClassifier 의 정답율 :  0.9777777777777777
# BernoulliNB 의 정답율 :  0.35555555555555557
# CalibratedClassifierCV 의 정답율 :  0.8222222222222222
# CategoricalNB 의 정답율 :  0.35555555555555557
# ComplementNB 의 정답율 :  0.6
# DecisionTreeClassifier 의 정답율 :  0.9555555555555556
# DummyClassifier 의 정답율 :  0.28888888888888886
# ExtraTreeClassifier 의 정답율 :  0.9111111111111111
# ExtraTreesClassifier 의 정답율 :  0.9555555555555556
# GaussianNB 의 정답율 :  0.9777777777777777
# GaussianProcessClassifier 의 정답율 :  0.9111111111111111
# GradientBoostingClassifier 의 정답율 :  0.9777777777777777
# HistGradientBoostingClassifier 의 정답율 :  0.9555555555555556
# KNeighborsClassifier 의 정답율 :  0.9555555555555556
# LabelPropagation 의 정답율 :  0.9555555555555556
# LabelSpreading 의 정답율 :  0.9555555555555556
# LinearDiscriminantAnalysis 의 정답율 :  0.9777777777777777
# LinearSVC 의 정답율 :  0.8444444444444444
# LogisticRegression 의 정답율 :  0.9111111111111111
# LogisticRegressionCV 의 정답율 :  0.9777777777777777
# MLPClassifier 의 정답율 :  0.9333333333333333
# MultinomialNB 의 정답율 :  0.6
# NearestCentroid 의 정답율 :  0.9555555555555556
# NuSVC 의 정답율 :  0.9777777777777777
# PassiveAggressiveClassifier 의 정답율 :  0.8888888888888888
# Perceptron 의 정답율 :  0.5555555555555556
# QuadraticDiscriminantAnalysis 의 정답율 :  0.9777777777777777
# RadiusNeighborsClassifier 의 정답율 :  0.6
# RandomForestClassifier 의 정답율 :  0.9777777777777777
# RidgeClassifier 의 정답율 :  0.7111111111111111
# RidgeClassifierCV 의 정답율 :  0.7333333333333333
# SGDClassifier 의 정답율 :  0.9333333333333333
# SVC 의 정답율 :  0.9777777777777777