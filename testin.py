import numpy as np

nsig =4
nthet=5
local_size1=3
x = np.zeros((nsig,nthet,local_size1))
x[:,:,:] = np.arange(nsig*nthet*local_size1).reshape(nsig,nthet,local_size1)
print(x.flatten())
idx1=2
idx2=3
idx3=1
x[idx1,idx2,idx3] = 999
print(x[:,:,0])
print(x[:,:,1])
print(x[:,:,2])
idxflat = idx1*(nthet*local_size1) + idx2*(local_size1) + idx3

print('flattened index',idxflat)
#y=np.array([1,2,3,4])
#x= np.multiply(x.T,y).T
print(x[:,:,0])
print(x[:,:,1])
print(x[:,:,2])

print('flattened version')
xflat=x.flatten()
print(xflat)
print('indexed flat',xflat[idxflat])
b = np.array([710,720,50,20])
a =np.minimum(710,b)
x = np.sinh(a)
print('npsinh',1/x)
