import numpy as np 
from sklearn.datasets import load_iris


#1.데이터 
datasets = load_iris()
x = datasets.data 
y = datasets.target 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True)

#2.모델구성 
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor 
from xgboost import XGBClassifier,XGBRegressor

model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()

#3.훈련 
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4.평가, 예측 
# result = model.score(x_test,y_test)
# print("model.score : ", result)

from sklearn.metrics import r2_score
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print('r2_score : ', r2)/

print('================================================')
print(model1,':',model1.feature_importances_) # 열의 중요도를 나타냄. 필요없는 컬럼 뻬기위함. 

import matplotlib.pyplot as plt 



def plot_feature_importances(model): 
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align= 'center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    if model == DecisionTreeRegressor: 
        plt.title(model)
    
plt.subplot(2, 2, 1) 
plt.title('Decision')   
plot_feature_importances(model1)

plt.subplot(2, 2, 2)
plt.title('Random') 
plot_feature_importances(model2)

plt.subplot(2, 2, 3)  
plt.title('Gradient') 
plot_feature_importances(model3)

plt.subplot(2, 2, 4)   
plt.title('xgb')
plot_feature_importances(model4)
        
plt.show()


