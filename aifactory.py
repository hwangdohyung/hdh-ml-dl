import pandas as pd 
import numpy as np 

path = 'D:/aifactory/'
data = pd.read_csv(path + '월별공급량및비중.csv')



def split_x(data, y1):
    aaa= [ ]
    for i in range(len(data) - size +1): 
        subset = data[i : (i + size)] 
        aaa.append(subset)   
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) 

x= bbb[:, :-1]
y= bbb[:, -1]
print(x,y)

print(x.shape, y.shape) 

x = x.reshape(96,4,1)

####################################### predict ####################################
size1= 4
def predict_x(data, y1):
    ccc= [ ]
    for i in range(len(data) - size1 +1): 
        subset = data[i : (i + size1)] 
        ccc.append(subset)   
    return np.array(ccc)

ddd = predict_x(x_predict, size1)


x_pred= ddd[:, :]