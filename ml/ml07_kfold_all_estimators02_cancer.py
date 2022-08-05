from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']
print(x.shape, y.shape)

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

# AdaBoostClassifier 의 정답율 :  [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133]
# BaggingClassifier 의 정답율 :  [0.92105263 0.93859649 0.93859649 0.92982456 0.95575221]
# BernoulliNB 의 정답율 :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
# CalibratedClassifierCV 의 정답율 :  [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133]
# CategoricalNB 의 정답율 :  [nan nan nan nan nan]
# ComplementNB 의 정답율 :  [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531]
# DecisionTreeClassifier 의 정답율 :  [0.92982456 0.93859649 0.9122807  0.87719298 0.95575221]
# DummyClassifier 의 정답율 :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
# ExtraTreeClassifier 의 정답율 :  [0.92982456 0.92105263 0.92105263 0.85087719 0.94690265]
# ExtraTreesClassifier 의 정답율 :  [0.96491228 0.98245614 0.96491228 0.94736842 0.99115044]
# GaussianNB 의 정답율 :  [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221]
# GaussianProcessClassifier 의 정답율 :  [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265]
# GradientBoostingClassifier 의 정답율 :  [0.94736842 0.96491228 0.95614035 0.93859649 0.98230088]
# HistGradientBoostingClassifier 의 정답율 :  [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088]
# KNeighborsClassifier 의 정답율 :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
# LabelPropagation 의 정답율 :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
# LabelSpreading 의 정답율 :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
# LinearDiscriminantAnalysis 의 정답율 :  [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133]
# LinearSVC 의 정답율 :  [0.80701754 0.92105263 0.83333333 0.85087719 0.85840708]
# LogisticRegression 의 정답율 :  [0.93859649 0.95614035 0.88596491 0.95614035 0.95575221]
# LogisticRegressionCV 의 정답율 :  [0.95614035 0.97368421 0.9122807  0.96491228 0.96460177]
# MLPClassifier 의 정답율 :  [0.89473684 0.93859649 0.90350877 0.93859649 0.97345133]
# MultinomialNB 의 정답율 :  [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531]
# NearestCentroid 의 정답율 :  [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442]
# NuSVC 의 정답율 :  [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575]
# PassiveAggressiveClassifier 의 정답율 :  [0.81578947 0.93859649 0.88596491 0.90350877 0.86725664]
# Perceptron 의 정답율 :  [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265]
# QuadraticDiscriminantAnalysis 의 정답율 :  [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265]
# RadiusNeighborsClassifier 의 정답율 :  [nan nan nan nan nan]
# RandomForestClassifier 의 정답율 :  [0.96491228 0.96491228 0.96491228 0.95614035 0.98230088]
# RidgeClassifier 의 정답율 :  [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221]
# RidgeClassifierCV 의 정답율 :  [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177]
# SGDClassifier 의 정답율 :  [0.90350877 0.85087719 0.88596491 0.90350877 0.66371681]
# SVC 의 정답율 :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]