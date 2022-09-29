 
from tkinter import E
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

############################## DATA LOADER ################################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train,y_train)  # x, y 를 합쳐준다.
test_set = TensorDataset(x_test,y_test) 

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set,batch_size=32, shuffle=False)


#2.모델 

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

# model = Model(30, 1).to(DEVICE)
              
############## load model ################
model2 = Model(30,1).to(DEVICE)

path = 'D:\study_data\_save/'
# torch.save(model.state_dict(), path + 'torch13_state_dict.pt')

model2.load_state_dict(torch.load(path + 'torch13_state_dict.pt'))

###########################################

y_predict = (model2(x_test) >= 0.5).float()
print(y_predict[:10])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test,y_predict)      # 에러 
# print('accuracy_score : ', score) 

score = accuracy_score(y_test.cpu(),y_predict.cpu())
print('accuracy : ', score)



# # 3.컴파일,훈련 
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(),lr=0.01)

# def train(model,criterion,optimizer,loader):
    
#     total_loss = 0
#     for x_batch, y_batch in loader:
#         optimizer.zero_grad()
#         hypothesis = model(x_batch)
#         loss = criterion(hypothesis,y_batch)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#     return total_loss / len(loader)


# epochs = 100
# for epoch in range(epochs+1):
#     loss = train(model,criterion,optimizer,train_loader)
#     print('epochs : {}, loss : {}'.format(epochs,loss))


# #4.평가,예측
# def evaluate(model,criterion,loader):
#     model.eval()
#     total_loss = 0
    
#     for x_batch,y_batch in loader:
    
#         with torch.no_grad():
#             y_predict = model(x_batch)
#             loss = criterion(y_predict,y_batch)
#             total_loss += loss.item()
            
#     return total_loss

# loss2 = evaluate(model,criterion,test_loader)
# print('최종 loss : ', loss2)

# y_predict = (model(x_test) >= 0.5).float()
# print(y_predict[:10])

# score = (y_predict == y_test).float().mean()
# print('accuracy : {:.4f}'.format(score))

# from sklearn.metrics import accuracy_score
# # score = accuracy_score(y_test,y_predict)      # 에러 
# # print('accuracy_score : ', score) 

# score = accuracy_score(y_test.cpu(),y_predict.cpu())
# print('accuracy : ', score)

# path = 'D:\study_data\_save/'
# torch.save(model.state_dict(), path + 'torch13_state_dict.pt')
