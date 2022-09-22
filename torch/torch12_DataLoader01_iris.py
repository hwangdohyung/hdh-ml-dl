#회귀모델에 시그모이드를 붙인 모델 logistic regression 2진분류에서만 사용 

from sklearn.datasets import load_iris
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1.데이터 
datasets = load_iris()
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

#############################
from torch.utils.data import TensorDataset,DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(test_set,batch_size=64,shuffle=False)

#2.모델 
# model = nn.Sequential(
#     nn.Linear(4,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,3),
#     nn.Softmax(),      # softmax 안써도 아래 loss에서 처리해줌.(nn.CrossEntropyLoss)
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
        nn.Softmax()
        )
        
    def forward(self, input_size):
        x = self.net(input_size)
        return x
    
model = Model(4, 3).to(DEVICE)


#3.컴파일,훈련 
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,loader):
    total_loss=0
    
    for x_batch,y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

epochs = 1000
for epoch in range(epochs+1):
    loss = train(model,criterion,optimizer,train_loader)
    if epoch % 100 == 0:
        print('epochs : {}, loss : {}'.format(epochs,loss))
    
#4.평가,예측
def evaluate(model,criterion,loader):
    total_loss = 0
    model.eval()
    
    for x_batch,y_batch in loader:
        
        with torch.no_grad():
            y_predict = model(x_batch)
            results = criterion(y_predict,y_batch)    
            total_loss += results.item()
            
    return total_loss

loss2 = evaluate(model,criterion,test_loader)
print('최종 loss : ', loss2)

y_predict = torch.argmax(model(x_test),axis=1)
from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(y_predict.cpu(),y_test.cpu()))


# 최종 loss :  0.6443460583686829
# acc :  0.9
