import numpy as np

x = np.zeros((4,3))
x[:,:] = np.arange(12).reshape(4,3)
print(x)

y=np.array([1,2,3,4])
x= np.multiply(x.T,y).T
print(x)
print(x.flatten())

x=6
y=x

x=10
print(y)

y = np.arange(50)
print(len(y[4:49]))
