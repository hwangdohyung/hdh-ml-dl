import numpy as np 
import matplotlib.pyplot as plt

a=1 

def elu(x):
  if x>0:
    return x
  else:
    return a*(np.exp(x)-1)

x = np.arange(-5, 5, 0.1)
y = np.array([elu(x) for x in x])

plt.plot(x, y)
plt.title('ELU')
plt.grid()
plt.ylim([-3.0, 3.0])
plt.xlim([-3.0, 3.0])
plt.show()





