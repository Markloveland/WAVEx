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



local_dof_coords1 = np.zeros((31,2))
local_dof_coords1[:,1] = np.arange(0,30,31)
depth = np.zeros(31)
print('bathymetry locations')
bath_locs = np.linspace(0,4.4*7,8)
bath_vals = np.array([0.7,0.7,0.64,0.424,0.208,0.315,0.124,-0.06])
tol =1e-8
for a in range(1,bath_locs.shape[0]):
    seg = np.logical_and(local_dof_coords1[:,1]>=bath_locs[a-1]-tol,local_dof_coords1[:,1]<=bath_locs[a]+tol)
    depth[seg] = (bath_vals[a] - bath_vals[a-1])/(bath_locs[a]-bath_locs[a-1])*(local_dof_coords1[seg,1]-bath_locs[a-1]) + bath_vals[a-1]
'''
#repeat for water depth and add
wlev_locs = np.linspace(0,30,31)
wlev_vals = np.zeros(wlev_locs.shape)
wlev_vals[:13] = 0.062
wlev_vals[13:16] = 0.061
wlev_vals[16:18] = 0.060
wlev_vals[18] = 0.061
wlev_vals[19] = 0.062
wlev_vals[20] = 0.061
wlev_vals[21:23] = 0.062
wlev_vals[23:25] = 0.061
wlev_vals[25] = 0.060
wlev_vals[26] = 0.061
wlev_vals[27] = 0.066
wlev_vals[28:] = 0.067
for a in range(1,wlev_locs.shape[0]):
    seg = np.logical_and(local_dof_coords1[:,1]>=wlev_locs[a-1]-tol,local_dof_coords1[:,1]<=wlev_locs[a]+tol)
    depth[seg] = depth[seg] + (wlev_vals[a] - wlev_vals[a-1])/(wlev_locs[a]-wlev_locs[a-1])*(local_dof_coords1[seg,1]-wlev_locs[a-1]) + wlev_vals[a-1]
'''







a=2
seg = np.logical_and(local_dof_coords1[:,1]>=bath_locs[a-1],local_dof_coords1[:,1]<=bath_locs[a])
print(local_dof_coords1)
print(local_dof_coords1[seg,1])


dum = np.ones((5,1))
dum2 = 5*np.ones(5)

dum[:,0] = dum2.flatten()[:]

print(dum)
