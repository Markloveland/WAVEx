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


print('np where')
x = np.arange(12)
print('roriginal matrix',x)
local_size2=3
x[2]=0
tol = 9
limit_idx = np.where(x>tol)[0]
print('imit_idx',limit_idx)
if np.any(limit_idx):
    temp = np.unique( np.floor(limit_idx/local_size2) )
    locs = np.kron(temp,np.ones(local_size2))
    dum = np.kron(np.ones(temp.shape),np.arange(local_size2))
    idx2 = np.array(dum + locs*local_size2,dtype=np.int32)

    print('index to shut down',idx2)


print('bathymetry locations')
bath_locs = np.linspace(4.4*1-7.4,4.4*7-7.4,7)
bath_vals = np.array([0.7,0.64,0.424,0.208,0.315,0.124,-0.06])
print('bath_locs',bath_locs)
local_dof_coords1 = np.zeros((31,2))
local_dof_coords1[:,1] = np.arange(0,31)
local_dof_coords1[1:2,0] = 1
a=2
seg = np.logical_and(local_dof_coords1[:,1]>=bath_locs[a-1],local_dof_coords1[:,1]<=bath_locs[a])
print(local_dof_coords1)
print(local_dof_coords1[seg,1])

