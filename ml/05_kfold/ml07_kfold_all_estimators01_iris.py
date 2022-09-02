from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_iris()

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

# AdaBoostClassifier 의 정답율 :  [0.63333333 0.93333333 1.         0.9        0.96666667]
# BaggingClassifier 의 정답율 :  [0.96666667 0.93333333 1.         0.9        0.96666667]
# BernoulliNB 의 정답율 :  [0.3        0.33333333 0.3        0.23333333 0.3       ]
# CalibratedClassifierCV 의 정답율 :  [0.9        0.83333333 1.         0.86666667 0.96666667]
# CategoricalNB 의 정답율 :  [0.9        0.93333333 0.93333333 0.9        1.        ]
# ComplementNB 의 정답율 :  [0.66666667 0.66666667 0.7        0.6        0.7       ]
# DecisionTreeClassifier 의 정답율 :  [0.93333333 0.96666667 1.         0.9        0.93333333]
# DummyClassifier 의 정답율 :  [0.3        0.33333333 0.3        0.23333333 0.3       ]
# ExtraTreeClassifier 의 정답율 :  [0.86666667 0.96666667 1.         0.9        0.9       ]
# ExtraTreesClassifier 의 정답율 :  [0.96666667 0.96666667 1.         0.86666667 0.96666667]
# GaussianNB 의 정답율 :  [0.96666667 0.9        1.         0.9        0.96666667]
# GaussianProcessClassifier 의 정답율 :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# GradientBoostingClassifier 의 정답율 :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# HistGradientBoostingClassifier 의 정답율 :  [0.86666667 0.96666667 1.         0.9        0.96666667]
# KNeighborsClassifier 의 정답율 :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# LabelPropagation 의 정답율 :  [0.93333333 1.         1.         0.9        0.96666667]
# LabelSpreading 의 정답율 :  [0.93333333 1.         1.         0.9        0.96666667]
# LinearDiscriminantAnalysis 의 정답율 :  [1.  1.  1.  0.9 1. ]
# LinearSVC 의 정답율 :  [0.96666667 0.96666667 1.         0.9        1.        ]
# LogisticRegression 의 정답율 :  [1.         0.96666667 1.         0.9        0.96666667]
# LogisticRegressionCV 의 정답율 :  [1.         0.96666667 1.         0.9        1.        ]
# MLPClassifier 의 정답율 :  [0.96666667 0.96666667 1.         0.93333333 1.        ]
# MultinomialNB 의 정답율 :  [0.96666667 0.93333333 1.         0.93333333 1.        ]
# NearestCentroid 의 정답율 :  [0.93333333 0.9        0.96666667 0.9        0.96666667]
# NuSVC 의 정답율 :  [0.96666667 0.96666667 1.         0.93333333 1.        ]
# PassiveAggressiveClassifier 의 정답율 :  [0.9        0.96666667 0.8        0.6        0.93333333]
# Perceptron 의 정답율 :  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ]
# QuadraticDiscriminantAnalysis 의 정답율 :  [1.         0.96666667 1.         0.93333333 1.        ]
# RadiusNeighborsClassifier 의 정답율 :  [0.96666667 0.9        0.96666667 0.93333333 1.        ]
# RandomForestClassifier 의 정답율 :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# RidgeClassifier 의 정답율 :  [0.86666667 0.8        0.93333333 0.7        0.9       ]
# RidgeClassifierCV 의 정답율 :  [0.86666667 0.8        0.93333333 0.7        0.9       ]
# SGDClassifier 의 정답율 :  [0.63333333 0.76666667 0.73333333 0.6        0.9       ]
# SVC 의 정답율 :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]