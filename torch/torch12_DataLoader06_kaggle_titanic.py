#[실습]#
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1.데이터
path = 'D:\study_data\_data\kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',index_col =0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)
##########전처리############
train_test_data = [train_set, test_set]
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(dataset)

for dataset in train_test_data:
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    
    # 가족수 > 1이면 동승자 있음
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {'S':0, 'C':1, 'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4)

train_set['Cabin'] = train_set['Cabin'].str[:1]
for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4])
for dataset in train_test_data:
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test_data:
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])
    drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    dataset = dataset.drop(drop_column, axis=1, inplace=True)
print(train_set.head())


x = train_set.drop(['Survived'], axis=1,)
y = train_set['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2,random_state=61,stratify=y)

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape,test_set.shape)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
test_set = torch.FloatTensor(test_set).to(DEVICE)

####################################################
from torch.utils.data import TensorDataset,DataLoader
train_set = TensorDataset(x_train,y_train)
test_set1 = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_set, batch_size=64,shuffle=True)
test_loader = DataLoader(test_set1, batch_size=64,shuffle=False)

#2.모델 
# model = nn.Sequential(
#     nn.Linear(8,10),
#     nn.ReLU(),
#     nn.Linear(10,8),
#     nn.ReLU(),
#     nn.Linear(8,6),
#     nn.ReLU(),
#     nn.Linear(6,4),
#     nn.ReLU(),
#     nn.Linear(4,1),
#     nn.Sigmoid(),
# ).to(DEVICE)


class Model(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.L1 = nn.Linear(input_dim, 10)
        self.L2 = nn.Linear(10,8)
        self.L3 = nn.Linear(8,6)
        self.L4 = nn.Linear(6,4)
        self.L5 = nn.Linear(4,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,input_size):
        x = self.L1(input_size)
        x = self.relu(x)       
        x = self.L2(x)
        x = self.relu(x)       
        x = self.L3(x)
        x = self.relu(x)       
        x = self.L4(x)
        x = self.relu(x)       
        x = self.L5(x)
        x = self.sigmoid(x)
        return x

model = Model(8,1).to(DEVICE)

#3.컴파일,훈련 
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model,criterion, optimizer,loader):
    total_loss = 0 
    for x_batch,y_batch in loader:
            
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

epochs = 1000
for epoch in range(epochs+1):
    loss = train(model,criterion,optimizer,train_loader)
    print('epochs : {}, loss : {}'.format(epochs,loss))
    
    
#4.평가,예측 
def evaluate(model,criterion,loader):
    model.eval()
    total_loss = 0
    for x_batch,y_batch in loader:
        
        with torch.no_grad():
        
            y_predict = model(x_batch)
    
            results = criterion(y_predict,y_batch)
            total_loss += results.item()
            
        return total_loss

loss2 = evaluate(model,criterion,test_loader)
print('loss2 : ', loss2)

y_predict = (model(x_test) >= 0.5).float()

submit = (model(test_set) >= 0.5).float()

score = (y_predict == y_test).float().mean()
print('accuracy : ', score)
score = accuracy_score(y_predict.cpu(), y_test.cpu())
print('accuracy : ', score)

# accuracy :  tensor(0.8045, device='cuda:0')
# accuracy :  0.8044692737430168

