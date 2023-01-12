import numpy as np

x = np.zeros((4,3,3))
x[:,:,:] = np.arange(36).reshape(4,3,3)
print(x[:,:,0])
print(x[:,:,1])
print(x[:,:,2])

y=np.array([1,2,3,4])
x= np.multiply(x.T,y).T
print(x[:,:,0])
print(x[:,:,1])
print(x[:,:,2])

b = np.array([710,720,50,20])
a =np.minimum(710,b)
x = np.sinh(a)
print('npsinh',1/x)
