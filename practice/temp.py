import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


#1.데이터 
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],     
              [9,8,7,6,5,4,3,2,1,0]])  
predict = np.array([9])

x = x.T
y = y.T

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).to(DEVICE)
predict = torch.FloatTensor(predict).to(DEVICE)

#스케일링# 
predict = (predict - torch.mean(x)) / torch.std(x)
x = (x - torch.mean(x)) / torch.std(x)

#모델# 
model = nn.Sequential(
    nn.Linear(1,10),
    nn.ReLU(),
    nn.Linear(10,8),
    nn.ReLU(),
    nn.Linear(8,6),
    nn.ReLU(),
    nn.Linear(6,3)
).to(DEVICE)

#컴파일,훈련 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y)

    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print('epochs : {}, loss : {}'.format(epochs,loss))

#평가,예측 
def evaluate(model,criterion,x,y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        results = criterion(y_predict,y)
        
    return results.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

results = model(predict)
results = results.tolist()
print('predict의 결과 : ',results)









