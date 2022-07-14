import numpy as np 

a = np.array(range(1,11))
size = 5

def split_x(x, y):
    aaa= [ ]
    for i in range(len(x) - size +1): 
        subset = x[i : (i + size)] 
        aaa.append(subset)   
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) #(6,5)

x1= bbb[:, :-1]
y1= bbb[:, -1]
print(x1,y1)
print(x1.shape, y1.shape) #(6,4) (6,)

# 과제 분석 