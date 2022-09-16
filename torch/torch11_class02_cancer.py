#회귀모델에 시그모이드를 붙인 모델 logistic regression 2진분류에서만 사용 

from sklearn.datasets import load_breast_cancer
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1.데이터 
datasets = load_breast_cancer()
x,y = datasets.data,datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#2.모델 
# model = nn.Sequential(
#     nn.Linear(30,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
#     nn.Sigmoid(),
# ).to(DEVICE)

##################### 클래스화 ####################
class Model(nn.Module):            # nn.module 이란 클래스를 상속받겠다.
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # super(Model,self).__init__() # 위와 동일 
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_size):  
        x = self.linear1(input_size) 
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x) 
        x = self.linear3(x) 
        x = self.relu(x) 
        x = self.linear4(x) 
        x = self.sigmoid(x) 
        return x 

model = Model(30, 1).to(DEVICE)
              

#3.컴파일,훈련 
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis,y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 1000
for epoch in range(epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
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

y_predict = (model(x_test) >= 0.5).float()
print(y_predict[:10])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test,y_predict)      # 에러 
# print('accuracy_score : ', score) 

score = accuracy_score(y_test.cpu(),y_predict.cpu())
print('accuracy : ', score)

# accuracy : 0.9766
# accuracy :  0.9766081871345029


