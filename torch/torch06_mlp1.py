### 데이터와 모델만 바꿔준다 ###

from inspect import Parameter
from pickletools import optimize
from unittest import result
import numpy as np 
import torch 
print(torch.__version__) #1.12.1

import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')  # 쓸 수 있는 장치가 있으면 cuda 없으면 cpu를 쓰겟다
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)


# 1.데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])   
y = np.array([11,12,13,14,15,16,17,18,19,20])   
x_test = np.array([10, 1.3])

x = torch.FloatTensor(x).to(DEVICE)  
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)


x = x.T
x_test = x_test.T

# # 스케일링 #
x_test = (x_test - torch.mean(x)) / torch.std(x)
x = (x - torch.mean(x)) / torch.std(x) # standard scaler


print(x,y)
print(x.shape,y.shape)

# 2.모델
# model = Sequential()
model = nn.Sequential(
    nn.Linear(2,6),
    nn.Linear(6,5),
    nn.Linear(5,3),
    nn.ReLU(),
    nn.Linear(3,2),
    nn.Linear(2,1),
    ).to(DEVICE)

# 3.컴파일,훈련 
# model.compile(loss='mse',optimizer='SGD')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)
# optimizer = optim.Adam(model.parameters(), lr = 0.001)


def train(model, criterion, optimizer, x, y):
    # model.train()         # 훈련모드 *생략해도 됨(default)
    optimizer.zero_grad()    # 손실함수의 기울기를 초기화한다. 역전파할때 미분값이 남아있지 않게 하기위해  #1 외우기 
    
    hypothesis = model(x)    # model에 x를 넣었을 때 h 반환
    
    loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y)  
    # loss = F.mse_loss(hypothesis, y)
    
    loss.backward()                                          #2 외우기
    optimizer.step()                                         #3 외우기
    return loss.item()


epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epochs : {}, loss: {}'.format(epoch,loss))
    
    
# 4.평가, 예측
# loss = model.evaluate(x,y)

def evaluate(model, criterion, x, y):
    model.eval()                # 평가모드 
    
    with torch.no_grad():    
        y_predict = model(x)
        results = criterion(y_predict,y)
    return results.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)
print(x_test)


results = model(torch.Tensor(x_test).to(DEVICE)) 

print('x_test의 예측값 : ', results.item())


# 최종 loss :  0.0002392903988948092
# tensor([[8.8575, 0.1575]], device='cuda:0')
# x_test의 예측값 :  19.97201156616211






