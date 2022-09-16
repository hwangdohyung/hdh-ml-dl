#회귀모델에 시그모이드를 붙인 모델 logistic regression 2진분류에서만 사용 

from sklearn.datasets import fetch_covtype
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1.데이터 
datasets = fetch_covtype()
x,y = datasets.data,datasets.target

x = torch.FloatTensor(x)
y = torch.LongTensor(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,shuffle=True, random_state=123, stratify=y)


x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)


#2.모델 
# model = nn.Sequential(
#     nn.Linear(8,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,7),
#     nn.Softmax(),      # softmax 안써도 아래 loss에서 처리해줌.(nn.CrossEntropyLoss)
# ).to(DEVICE)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.L1 = nn.Linear(input_dim, 64)    
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 16)
        self.L4 = nn.Linear(16, 7)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, input_size):
        x = self.L1(input_size)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        x = self.relu(x)
        x = self.L4(x)
        x = self.softmax(x)
        return x
model = Model(54, 7).to(DEVICE)


#3.컴파일,훈련 
criterion = nn.CrossEntropyLoss()  
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
from sklearn.metrics import accuracy_score
y_predict = model(x_test)
y_predict= torch.argmax(y_predict,axis=1)
score = accuracy_score(y_predict.cpu(),y_test.cpu())
print(score)


