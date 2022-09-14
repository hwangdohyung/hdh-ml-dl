from inspect import Parameter
from pickletools import optimize
from unittest import result
import numpy as np 
import torch 
print(torch.__version__) #1.12.1

import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F


# 1.데이터
x = np.array([1,2,3])   # (3, )
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)  # 1번째 자리에 차원 늘려줌. (3, ) -> (3, 1)
y = torch.FloatTensor(y).reshape(3,1)

print(x.shape,y.shape)

# 2.모델
# model = Sequential()
model = nn.Linear(1, 1) # (인풋 x값 , 아웃풋 y값)

# 3.컴파일,훈련 
# model.compile(loss='mse',optimizer='SGD')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
# optim.Adam(model.parameters(), lr = 0.01)


def train(model, creiterion, optimizer, x, y):
    # model.train()         # 훈련모드 *생략해도 됨(default)
    optimizer.zero_grad()    # 손실함수의 기울기를 초기화한다. 역전파할때 미분값이 남아있지 않게 하기위해  #1 외우기 
    
    hypothesis = model(x)    # model에 x를 넣었을 때 h 반환
    
    loss = creiterion(hypothesis,y)  # mse에 h와 y 넣겠다.
    
    
    loss.backward()                                          #2 외우기
    optimizer.step()                                         #3 외우기
    return loss.item()


epochs = 1000
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

# y_predict = model.predict([4])

results = model(torch.Tensor([[4]])) #2차원으로 넣어줘야함.

print('4의 예측값 : ', results.item())










