#n_component> 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것!

#m27_2 결과를 뛰어넘어라!!!

# parameters = [
#     {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
#      "max_depth":[4,5,6]},
#     {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
#      "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
#     {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.5],
#      "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
#      "colsample_bylevel":[0.6, 0.7, 0.9]}
# ]
# n_jobs = -1 
#     tree_method = 'gpu_hist',
#     predictor = 'gpu_predictor',
#     gpu_id = 0,

# 실습 시작!

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import warnings 
warnings.filterwarnings(action='ignore')
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#1. 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# PCA
x = np.append(x_train, x_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻
y = np.append(y_train, y_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻
x = x.reshape(70000, 28*28)


lda = LinearDiscriminantAnalysis(n_components=8) 
lda.fit(x, y)
x = lda.transform(x)

x_train = x[:60000]
x_test = x[60000:]

# parameter
parameter = [
    {'xg__n_estimator':[100, 200, 300], 'xg__learning_rate':[0.1, 0.3, 0.001, 0.01],
     'xg__max_depth':[4, 5, 6]},
    {'xg__n_estimator':[90, 100, 110], 'xg__learning_rate':[0.1, 0.001, 0.01],
     'xg__max_depth':[4, 5, 6], 'xg__colsample_bytree':[0.6, 0.9, 1]},
]

#2. 모델 구성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('mm', MinMaxScaler()), ('xg', XGBClassifier(tree_method = 'gpu_hist', predictor = 'gpu_predictor', gpu_id = 0,))],verbose=1)
model = RandomizedSearchCV(pipe, parameter, cv=5, verbose=1, n_jobs=1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print('걸린 시간 : ', round(end, 2))

#4. 평가, 예측
print("score : ", model.score(x_test, y_test))

#pca
# 걸린 시간 :  394.23
# score :  0.9659

#lda
# 걸린 시간 :  116.68
# score :  0.9121

