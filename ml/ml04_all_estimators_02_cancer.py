from matplotlib.pyplot import hist
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler


#1.데이터
datasets = load_breast_cancer()

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

# AdaBoostClassifier 의 정답율 :  0.9707602339181286
# BaggingClassifier 의 정답율 :  0.9649122807017544
# BernoulliNB 의 정답율 :  0.9239766081871345
# CalibratedClassifierCV 의 정답율 :  0.9532163742690059
# DecisionTreeClassifier 의 정답율 :  0.9590643274853801
# DummyClassifier 의 정답율 :  0.6374269005847953
# ExtraTreeClassifier 의 정답율 :  0.9298245614035088
# ExtraTreesClassifier 의 정답율 :  0.9590643274853801
# GaussianNB 의 정답율 :  0.9298245614035088
# GaussianProcessClassifier 의 정답율 :  0.9532163742690059
# GradientBoostingClassifier 의 정답율 :  0.9766081871345029
# HistGradientBoostingClassifier 의 정답율 :  0.9766081871345029
# KNeighborsClassifier 의 정답율 :  0.9590643274853801
# LabelPropagation 의 정답율 :  0.9532163742690059
# LabelSpreading 의 정답율 :  0.9532163742690059
# LinearDiscriminantAnalysis 의 정답율 :  0.9473684210526315
# LinearSVC 의 정답율 :  0.9707602339181286
# LogisticRegression 의 정답율 :  0.9824561403508771
# LogisticRegressionCV 의 정답율 :  0.9649122807017544
# MLPClassifier 의 정답율 :  0.9824561403508771
# NearestCentroid 의 정답율 :  0.9122807017543859
# NuSVC 의 정답율 :  0.935672514619883
# PassiveAggressiveClassifier 의 정답율 :  0.9707602339181286
# Perceptron 의 정답율 :  0.9766081871345029
# QuadraticDiscriminantAnalysis 의 정답율 :  0.935672514619883
# RandomForestClassifier 의 정답율 :  0.9649122807017544
# RidgeClassifier 의 정답율 :  0.9532163742690059
# RidgeClassifierCV 의 정답율 :  0.9590643274853801
# SGDClassifier 의 정답율 :  0.9707602339181286
# SVC 의 정답율 :  0.9707602339181286