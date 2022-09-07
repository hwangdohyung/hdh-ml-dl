import numpy as np 
import matplotlib.pyplot as plt

a=0.1

def leaky_relu(x):
  if x>0:
    return x
  else:
    return a*x

leaky_relu2 = lambda x : np.maximum(a*x, x)

x = np.arange(-5, 5, 0.1)
# y = np.array([leaky_relu(x) for x in x])
y = leaky_relu2(x)

plt.plot(x, y)
plt.title('Leaky_ReLU')
plt.ylim([-3.0, 3.0])
plt.xlim([-3.0, 3.0])
plt.grid()
plt.show()
