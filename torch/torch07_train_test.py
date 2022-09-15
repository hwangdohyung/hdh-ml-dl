from re import X
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# #1.데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
x_train = np.array([1,2,3,4,5,6,7])    #-훈련용 데이터           
x_test = np.array([8,9,10])            #-평가용 데이터
y_train = np.array([1,2,3,4,5,6,7])
y_test = np.array([8,9,10])
    
x_predict = np.array([11,12,13])

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
x_predict = torch.FloatTensor(x_predict).unsqueeze(1).to(DEVICE)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_predict.shape)

#2.모델구성
model = nn.Sequential(
    nn.Linear(1,10),
    nn.Linear(10,8),
    nn.Linear(8,6),
    nn.Linear(6,4),
    nn.Linear(4,1),
).to(DEVICE)

#3.컴파일,훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)
def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis,y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs : {}, loss : {}'.format(epochs,loss))
        
#4.평가,예측
def evaluate(model,criterion,x_test,y_test):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x_test)
        results = criterion(y_predict,y_test)
    return results.item()

loss2 = evaluate(model,criterion,x_test,y_test)
print('최종 loss : ', loss2)

results = model(x_predict).cpu().detach()
print('x_predict 결과 : ', results)

# 최종 loss :  6.063298192519884e-13
# x_predict 결과 :  tensor([[11.0000],[12.0000],[13.0000]])



