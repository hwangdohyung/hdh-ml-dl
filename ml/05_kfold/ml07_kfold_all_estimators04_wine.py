from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_wine()

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

# AdaBoostClassifier 의 정답율 :  [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857]
# BaggingClassifier 의 정답율 :  [1.         0.91666667 0.94444444 0.97142857 1.        ]
# BernoulliNB 의 정답율 :  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714]
# CalibratedClassifierCV 의 정답율 :  [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571]
# CategoricalNB 의 정답율 :  [       nan        nan        nan 0.94285714        nan]
# ComplementNB 의 정답율 :  [0.69444444 0.80555556 0.55555556 0.6        0.6       ]
# DecisionTreeClassifier 의 정답율 :  [0.91666667 0.97222222 0.91666667 0.85714286 0.94285714]
# DummyClassifier 의 정답율 :  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714]
# ExtraTreeClassifier 의 정답율 :  [0.88888889 0.83333333 0.88888889 0.88571429 0.85714286]
# ExtraTreesClassifier 의 정답율 :  [1.         0.97222222 1.         1.         1.        ]
# GaussianNB 의 정답율 :  [1.         0.91666667 0.97222222 0.97142857 1.        ]
# GaussianProcessClassifier 의 정답율 :  [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286]
# GradientBoostingClassifier 의 정답율 :  [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857]
# HistGradientBoostingClassifier 의 정답율 :  [0.97222222 0.94444444 1.         0.97142857 1.        ]
# KNeighborsClassifier 의 정답율 :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714]
# LabelPropagation 의 정답율 :  [0.52777778 0.47222222 0.5        0.4        0.54285714]
# LabelSpreading 의 정답율 :  [0.52777778 0.47222222 0.5        0.4        0.54285714]
# LinearDiscriminantAnalysis 의 정답율 :  [1.         0.97222222 1.         0.97142857 1.        ]
# LinearSVC 의 정답율 :  [0.86111111 0.75       0.66666667 0.82857143 0.94285714]
# LogisticRegression 의 정답율 :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ]
# LogisticRegressionCV 의 정답율 :  [0.97222222 0.91666667 0.97222222 0.94285714 0.97142857]
# MLPClassifier 의 정답율 :  [0.66666667 0.47222222 0.52777778 0.48571429 0.37142857]
# MultinomialNB 의 정답율 :  [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143]
# NearestCentroid 의 정답율 :  [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714]
# NuSVC 의 정답율 :  [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ]
# PassiveAggressiveClassifier 의 정답율 :  [0.69444444 0.77777778 0.61111111 0.62857143 0.6       ]
# Perceptron 의 정답율 :  [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143]
# QuadraticDiscriminantAnalysis 의 정답율 :  [0.97222222 1.         1.         1.         1.        ]
# RadiusNeighborsClassifier 의 정답율 :  [nan nan nan nan nan]
# RandomForestClassifier 의 정답율 :  [1.         0.94444444 1.         0.97142857 1.        ]
# RidgeClassifier 의 정답율 :  [1.         1.         1.         0.97142857 1.        ]
# RidgeClassifierCV 의 정답율 :  [1.         1.         1.         0.97142857 1.        ]
# SGDClassifier 의 정답율 :  [0.47222222 0.47222222 0.27777778 0.51428571 0.74285714]
# SVC 의 정답율 :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ]